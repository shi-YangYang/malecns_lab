import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Atlas } from '../../atlas/src/types';
import type { TelemetryBatch, TelemetryGroup } from './types';
import type { BrainClip, BrainDisplayData } from './brain-data';

export interface BrainLegendEntry { id: string; label: string; color: string; count: number }
const FRAME_MS = 1000 / 30;
const TRANSITION_MS = 650;
const TEXTURE_WIDTH = 512;
const PULSE_SECONDS = 1.2;
const MAX_PULSE_POINTS = 32; // Existing telemetry samples at most 32 neurons.

/** Original synapse positions and measured scalar overlays. Straight graph
 * edges join representative synapses, never reconstructed axon trajectories. */
export class BrainView {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(34, 1, .01, 100);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private pointGeometry = new THREE.BufferGeometry();
  private haloGeometry = new THREE.BufferGeometry();
  private lineGeometry = new THREE.BufferGeometry();
  private pulseGeometry = new THREE.BufferGeometry();
  private pulsePositions = new Float32Array(MAX_PULSE_POINTS * 3);
  private pulseTimes = new Float32Array(MAX_PULSE_POINTS);
  private pulseStrengths = new Float32Array(MAX_PULSE_POINTS);
  private pointMaterial: THREE.ShaderMaterial;
  private haloMaterial: THREE.ShaderMaterial;
  private lineMaterial: THREE.ShaderMaterial;
  private pulseMaterial: THREE.ShaderMaterial;
  private target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false });
  private compositeScene = new THREE.Scene();
  private compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private compositeMaterial: THREE.ShaderMaterial;
  private compositeGeometry = new THREE.PlaneGeometry(2, 2);
  private signal: Float32Array;
  private texture: THREE.DataTexture;
  private center = new THREE.Vector3();
  private radius = 1;
  private observer: ResizeObserver;
  private intersection: IntersectionObserver;
  private groups: { id: string; indices: number[] }[] = [];
  private bodyIndex: Map<string, number>;
  private somaIndex = new Map<string, number>();
  private firstSynapse: Int32Array;
  private pulseEvents = new Map<string, { position: number[]; firedAt: number; strength: number }>();
  private lastBatchIdentity = '';
  private lastBatchSequence = -1;
  private pulseUntil = 0;
  private frame = 0;
  private timer = 0;
  private pulseTimer = 0;
  private lastRender = 0;
  private transitionStart = -TRANSITION_MS;
  private inView = true;
  private disposed = false;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  private uniforms: { [name: string]: THREE.IUniform };
  private onVisibility = (): void => { if (document.hidden) this.cancelFrame(); else this.render(); };

  constructor(private container: HTMLElement, private atlas: Atlas | null, private data: BrainDisplayData) {
    const { positions, owners, centers, manifest: m } = data;
    this.bodyIndex = new Map(m.body_ids.map((id, i) => [id, i]));
    // Somas are usable only in the same exported coordinate system. Otherwise
    // an event is anchored to an original synapse of that exact neuron.
    const transform = (m as typeof m & { coordinate_transform?: unknown }).coordinate_transform;
    if (atlas && transform && JSON.stringify(atlas.coordinate_transform) === JSON.stringify(transform)) {
      this.somaIndex = new Map(atlas.body_ids.map((id, i) => [id, i]));
    }
    this.firstSynapse = new Int32Array(m.node_count).fill(-1);
    owners.forEach((owner, i) => { if (this.firstSynapse[owner] === -1) this.firstSynapse[owner] = i; });
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.domElement.dataset.testid = 'brain-canvas';
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.setAttribute('aria-label', 'MaleCNS 三维突触结构。拖动旋转，滚轮缩放，方向键旋转，Home 复位。连线为连接关系示意。');
    container.append(this.renderer.domElement);
    this.signal = new Float32Array(TEXTURE_WIDTH * Math.ceil(m.node_count / TEXTURE_WIDTH) * 4);
    this.texture = new THREE.DataTexture(this.signal, TEXTURE_WIDTH, Math.ceil(m.node_count / TEXTURE_WIDTH), THREE.RGBAFormat, THREE.FloatType);
    this.texture.needsUpdate = true;
    this.pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.pointGeometry.setAttribute('neuron', new THREE.Float32BufferAttribute(owners, 1));
    // A fixed subset of the same original coordinates supplies soft volume.
    // No generated surface, random points or inferred neurite paths are added.
    const haloCount = Math.ceil(owners.length / 8);
    const haloPositions = new Float32Array(haloCount * 3);
    const haloOwners = new Float32Array(haloCount);
    for (let i = 0; i < haloCount; i++) {
      haloPositions.set(positions.subarray(i * 24, i * 24 + 3), i * 3);
      haloOwners[i] = owners[i * 8];
    }
    this.haloGeometry.setAttribute('position', new THREE.BufferAttribute(haloPositions, 3));
    this.haloGeometry.setAttribute('neuron', new THREE.BufferAttribute(haloOwners, 1));
    this.pointGeometry.computeBoundingBox();
    this.pointGeometry.boundingBox!.getCenter(this.center);
    this.pointGeometry.computeBoundingSphere();
    this.radius = this.pointGeometry.boundingSphere!.radius;
    const linePositions = new Float32Array(m.edges.length * 6);
    const lineOwners = new Float32Array(m.edges.length * 2);
    m.edges.forEach(([pre, post], i) => {
      linePositions.set(centers.subarray(pre*3, pre*3+3), i*6);
      linePositions.set(centers.subarray(post*3, post*3+3), i*6+3);
      lineOwners[i*2] = pre; lineOwners[i*2+1] = post;
    });
    this.lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    this.lineGeometry.setAttribute('neuron', new THREE.BufferAttribute(lineOwners, 1));
    this.uniforms = {
      signal: { value: this.texture }, textureSize: { value: new THREE.Vector2(TEXTURE_WIDTH, Math.ceil(m.node_count / TEXTURE_WIDTH)) },
      blend: { value: 1 }, pixelRatio: { value: this.renderer.getPixelRatio() },
      radius: { value: this.radius }, cameraDistance: { value: 4 }, viewportScale: { value: 1 },
    };
    const signalShader = `
      attribute float neuron;
      uniform sampler2D signal;
      uniform vec2 textureSize;
      uniform float blend;
      float response() {
        vec2 uv = (vec2(mod(neuron, textureSize.x), floor(neuron / textureSize.x)) + 0.5) / textureSize;
        vec4 s = texture2D(signal, uv);
        return mix(s.r, s.g, smoothstep(0.0, 1.0, blend));
      }`;
    this.pointMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: `${signalShader}
        uniform float pixelRatio; uniform float radius; uniform float cameraDistance; uniform float viewportScale;
        varying float energy; varying float depth;
        void main() {
          energy = response();
          vec4 view = modelViewMatrix * vec4(position, 1.0);
          depth = clamp((cameraDistance + radius + view.z) / (2.0 * radius), 0.0, 1.0);
          gl_Position = projectionMatrix * view;
          gl_PointSize = (4.8 + sqrt(energy) * 2.2) * pixelRatio * viewportScale;
        }`,
      fragmentShader: `
        varying float energy; varying float depth;
        void main() {
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(p, p);
          if (r2 > 1.0) discard;
          float soft = exp(-4.5 * r2) - exp(-4.5);
          vec3 base = vec3(0.19, 0.39, 0.51);
          vec3 warm = vec3(1.0, 0.78, 0.43);
          vec3 color = mix(base, warm, smoothstep(0.02, 0.65, energy));
          float alpha = soft * (0.052 + energy * 0.26) * (0.45 + 0.55 * depth);
          gl_FragColor = vec4(color, alpha);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.haloMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: `${signalShader}
        uniform float pixelRatio; uniform float viewportScale;
        varying float energy;
        void main() {
          energy = response();
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (11.0 + sqrt(energy) * 13.0) * pixelRatio * viewportScale;
        }`,
      fragmentShader: `
        varying float energy;
        void main() {
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(p, p);
          if (r2 > 1.0) discard;
          float soft = exp(-5.0 * r2) - exp(-5.0);
          vec3 color = mix(vec3(0.12, 0.29, 0.42), vec3(1.0, 0.73, 0.36), smoothstep(0.02, 0.65, energy));
          gl_FragColor = vec4(color, soft * (0.009 + energy * 0.095));
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.lineMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: `${signalShader}
        varying float energy;
        void main() { energy = response(); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        varying float energy;
        void main() {
          gl_FragColor = vec4(mix(vec3(0.28, 0.48, 0.58), vec3(1.0, 0.80, 0.48), energy), 0.006 + 0.025 * energy);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.scene.add(new THREE.Points(this.haloGeometry, this.haloMaterial));
    this.scene.add(new THREE.Points(this.pointGeometry, this.pointMaterial));
    this.scene.add(new THREE.LineSegments(this.lineGeometry, this.lineMaterial));
    this.pulseGeometry.setAttribute('position', new THREE.BufferAttribute(this.pulsePositions, 3).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setAttribute('firedAt', new THREE.BufferAttribute(this.pulseTimes, 1).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setAttribute('strength', new THREE.BufferAttribute(this.pulseStrengths, 1).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setDrawRange(0, 0);
    this.pulseMaterial = new THREE.ShaderMaterial({
      uniforms: { time: { value: 0 }, pixelRatio: this.uniforms.pixelRatio, viewportScale: this.uniforms.viewportScale,
        reduced: { value: this.reducedMotion ? 1 : 0 } },
      vertexShader: `
        attribute float firedAt; attribute float strength;
        uniform float time; uniform float pixelRatio; uniform float viewportScale; uniform float reduced;
        varying float age; varying float intensity;
        void main() {
          age = (time - firedAt) / ${PULSE_SECONDS}; intensity = strength;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (23.0 + (1.0 - reduced) * 16.0 * clamp(age, 0.0, 1.0)) * pixelRatio * viewportScale;
        }`,
      fragmentShader: `
        uniform float reduced;
        varying float age; varying float intensity;
        void main() {
          if (age < 0.0 || age >= 1.0) discard;
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(p, p);
          if (r2 > 1.0) discard;
          float glow = exp(-5.0 * r2) - exp(-5.0);
          float core = exp(-30.0 * r2);
          float envelope = mix(pow(1.0 - age, 1.5), 1.0, reduced);
          gl_FragColor = vec4(vec3(1.0, 0.83, 0.51), (glow * 0.7 + core) * envelope * intensity);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const pulses = new THREE.Points(this.pulseGeometry, this.pulseMaterial);
    pulses.frustumCulled = false;
    this.scene.add(pulses);
    // Accumulate in linear floating point before mapping brightness. A small
    // screen-space reconstruction filter softens splats without adding anatomy.
    this.compositeMaterial = new THREE.ShaderMaterial({
      uniforms: { image: { value: this.target.texture }, texel: { value: new THREE.Vector2(1, 1) } },
      vertexShader: 'varying vec2 uvImage; void main(){ uvImage=uv; gl_Position=vec4(position.xy,0.0,1.0); }',
      fragmentShader: `
        uniform sampler2D image; uniform vec2 texel; varying vec2 uvImage;
        void main() {
          vec3 c = texture2D(image, uvImage).rgb * 0.40;
          c += texture2D(image, uvImage + vec2(texel.x, 0.0)).rgb * 0.15;
          c += texture2D(image, uvImage - vec2(texel.x, 0.0)).rgb * 0.15;
          c += texture2D(image, uvImage + vec2(0.0, texel.y)).rgb * 0.15;
          c += texture2D(image, uvImage - vec2(0.0, texel.y)).rgb * 0.15;
          gl_FragColor = vec4(vec3(1.0) - exp(-c * 1.25), 1.0);
          #include <colorspace_fragment>
        }`, depthTest: false, depthWrite: false,
    });
    this.compositeScene.add(new THREE.Mesh(this.compositeGeometry, this.compositeMaterial));
    // OrbitControls captures camera.up on construction, so set it first.
    this.camera.up.set(0, 0, -1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.minDistance = this.radius * .5;
    this.controls.maxDistance = this.radius * 10;
    this.controls.addEventListener('change', this.render);
    this.reset();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.intersection = new IntersectionObserver(entries => {
      this.inView = entries.some(e => e.isIntersecting);
      if (this.inView) this.render(); else this.cancelFrame();
    });
    this.intersection.observe(container);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.renderer.domElement.addEventListener('keydown', this.onKey);
    this.resize();
  }

  getLegend(): BrainLegendEntry[] { return [{ id: 'synapses', label: '突触结构', color: '#658d9e', count: this.data.manifest.point_count }]; }

  setGroups(groups: TelemetryGroup[]): void {
    this.groups = groups.map(g => ({ id: g.id, indices: g.body_ids.flatMap(id => { const i = this.bodyIndex.get(id); return i === undefined ? [] : [i]; }) }));
    this.clearPulses();
    this.setSignal(new Float32Array(this.data.manifest.node_count), true);
  }

  /** Group means color structure; only recorded spike_events create pulses. */
  update(rates: number[] | null, _groupId: string, maxRate: number, batch?: TelemetryBatch | null): void {
    const values = new Float32Array(this.data.manifest.node_count);
    const scale = Number.isFinite(maxRate) && maxRate > 0 ? maxRate : 1;
    this.groups.forEach((g, c) => {
      const rate = rates?.[c] ?? 0;
      const value = Number.isFinite(rate) ? Math.max(0, Math.min(1, rate / scale)) : 0;
      // The plot selection must not attenuate measured activity elsewhere.
      for (const i of g.indices) values[i] = Math.max(values[i], value);
    });
    if (rates === null || batch === null) this.clearPulses();
    else if (batch) this.updatePulses(batch, performance.now() / 1000);
    this.setSignal(values, rates === null);
    this.render();
  }

  /** Fixed common scale and actual L1/L2 order; action names do not affect color. */
  setRecordedFrame(clip: BrainClip | null, half = 0): void {
    this.clearPulses();
    const m = this.data.manifest;
    const values = new Float32Array(m.node_count);
    if (clip) {
      m.readout_indices.forEach((node, i) => {
        const baseline = m.reference_currents[half][i];
        const magnitude = Math.abs(clip.currents[half][i] - baseline) / m.display_scale_mv;
        // One shared display transfer curve, no action-specific masks or gains.
        values[node] = THREE.MathUtils.smoothstep(magnitude, .08, 1);
      });
    }
    this.setSignal(values, clip === null);
    this.render();
  }

  private clearPulses(): void {
    this.lastBatchIdentity = ''; this.lastBatchSequence = -1;
    this.pulseEvents.clear(); this.pulseUntil = 0;
    clearTimeout(this.pulseTimer); this.pulseTimer = 0;
    this.pulseGeometry.setDrawRange(0, 0);
  }

  private updatePulses(batch: TelemetryBatch, now: number): void {
    const identity = `${batch.run_id}:${batch.execution_id}`;
    if (identity === this.lastBatchIdentity && batch.batch_sequence <= this.lastBatchSequence) return;
    if (identity !== this.lastBatchIdentity) this.clearPulses();
    this.lastBatchIdentity = identity; this.lastBatchSequence = batch.batch_sequence;
    for (const [id, event] of this.pulseEvents) {
      if (now >= event.firedAt + PULSE_SECONDS) this.pulseEvents.delete(id);
    }
    const measured = new Map<string, { step: number; count: number }>();
    for (const event of batch.spike_events) {
      if (!Number.isFinite(event.step) || event.step < batch.step_start || event.step >= batch.step_end) continue;
      const previous = measured.get(event.body_id);
      measured.set(event.body_id, { step: Math.min(previous?.step ?? event.step, event.step), count: (previous?.count ?? 0) + 1 });
    }
    for (const [id, event] of measured) {
      const soma = this.somaIndex.get(id);
      const neuron = this.bodyIndex.get(id);
      const synapse = neuron === undefined ? -1 : this.firstSynapse[neuron];
      const position = soma !== undefined && this.atlas
        ? this.atlas.positions.slice(soma * 3, soma * 3 + 3)
        : synapse < 0 ? [] : Array.from(this.data.positions.subarray(synapse * 3, synapse * 3 + 3));
      if (position.length !== 3 || !position.every(Number.isFinite)) continue;
      if (!this.pulseEvents.has(id) && this.pulseEvents.size >= MAX_PULSE_POINTS) continue;
      const order = THREE.MathUtils.clamp((event.step - batch.step_start) / Math.max(1, batch.step_end - batch.step_start), 0, 1);
      // Recorded event order gets a short display envelope at one real point.
      // It is not an axonal propagation measurement or a simulated new spike.
      const firedAt = now + (this.reducedMotion ? 0 : order * .18);
      this.pulseEvents.set(id, { position, firedAt, strength: Math.min(1, .65 + Math.log2(event.count + 1) * .12) });
    }
    let count = 0;
    this.pulseUntil = 0;
    for (const event of this.pulseEvents.values()) {
      this.pulsePositions.set(event.position, count * 3);
      this.pulseTimes[count] = event.firedAt;
      this.pulseStrengths[count] = event.strength;
      this.pulseUntil = Math.max(this.pulseUntil, event.firedAt + PULSE_SECONDS);
      count++;
    }
    for (const name of ['position', 'firedAt', 'strength']) this.pulseGeometry.attributes[name].needsUpdate = true;
    this.pulseGeometry.setDrawRange(0, count);
  }

  private setSignal(values: Float32Array, immediate = false): void {
    if (!immediate && values.every((value, i) => value === this.signal[i * 4 + 1])) return;
    const progress = this.reducedMotion ? 1 : THREE.MathUtils.clamp((performance.now()-this.transitionStart)/TRANSITION_MS, 0, 1);
    const blend = progress * progress * (3-2*progress);
    for (let i = 0; i < values.length; i++) {
      this.signal[i*4] = immediate ? values[i] : this.signal[i*4] + (this.signal[i*4+1]-this.signal[i*4]) * blend;
      this.signal[i*4+1] = values[i];
    }
    this.texture.needsUpdate = true;
    this.transitionStart = performance.now() - (immediate ? TRANSITION_MS : 0);
    this.render();
  }

  private fitDistance(): number {
    const slope = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * .86;
    const inverse = this.camera.quaternion.clone().invert();
    const point = new THREE.Vector3();
    let distance = this.controls.minDistance;
    // The exact bounding box, not a fabricated anatomical envelope.
    const box = this.pointGeometry.boundingBox!;
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      point.set(x, y, z).sub(this.controls.target).applyQuaternion(inverse);
      distance = Math.max(distance, point.z + Math.max(Math.abs(point.x)/(slope*this.camera.aspect), Math.abs(point.y)/slope));
    }
    return distance;
  }

  reset(): void {
    this.controls.target.copy(this.center);
    // Original x is horizontal; z increases towards the ventral nerve cord.
    // No mirroring, anatomical warping or bilateral reconstruction.
    this.camera.up.set(0, 0, -1);
    this.camera.position.copy(this.center).add(new THREE.Vector3(0, -1, 0));
    this.camera.lookAt(this.center);
    this.camera.position.copy(this.center).add(new THREE.Vector3(0, -this.fitDistance(), 0));
    this.controls.update(); this.render();
  }

  private onKey = (event: KeyboardEvent): void => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', 'Home'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') { this.reset(); return; }
    const offset = this.camera.position.clone().sub(this.controls.target);
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') offset.applyAxisAngle(this.camera.up, event.key === 'ArrowLeft' ? .08 : -.08);
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') offset.applyAxisAngle(new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion), event.key === 'ArrowUp' ? .08 : -.08);
    if (event.key === '+' || event.key === '=') offset.multiplyScalar(.9);
    if (event.key === '-') offset.multiplyScalar(1.1);
    offset.clampLength(this.controls.minDistance, this.controls.maxDistance);
    this.camera.position.copy(this.controls.target).add(offset);
    this.controls.update(); this.render();
  };

  private resize(): void {
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    const previousFit = this.fitDistance();
    this.camera.aspect = width/height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    const pixels = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target.setSize(pixels.x, pixels.y);
    this.compositeMaterial.uniforms.texel.value.set(1 / pixels.x, 1 / pixels.y);
    this.uniforms.viewportScale.value = Math.max(.65, Math.min(1.4, height/900));
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.camera.position.copy(this.controls.target).add(offset.multiplyScalar(this.fitDistance()/previousFit));
    this.controls.update(); this.render();
  }

  private cancelFrame(): void {
    cancelAnimationFrame(this.frame); clearTimeout(this.timer); clearTimeout(this.pulseTimer);
    this.frame = 0; this.timer = 0; this.pulseTimer = 0;
  }
  private render = (): void => {
    if (this.disposed || this.frame || this.timer || document.hidden || !this.inView) return;
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      if (this.disposed || document.hidden || !this.inView) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        if (this.disposed || document.hidden || !this.inView) return;
        const now = performance.now();
        const progress = Math.min(1, (now-this.transitionStart)/TRANSITION_MS);
        this.pulseMaterial.uniforms.time.value = now / 1000;
        if (now / 1000 >= this.pulseUntil) this.pulseGeometry.setDrawRange(0, 0);
        this.uniforms.blend.value = this.reducedMotion ? 1 : progress;
        this.uniforms.cameraDistance.value = this.camera.position.distanceTo(this.center);
        this.renderer.setRenderTarget(this.target);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.compositeScene, this.compositeCamera);
        this.lastRender = performance.now();
        if (this.reducedMotion && now / 1000 < this.pulseUntil) {
          // Redraw static markers at each next expiry, without frame animation.
          clearTimeout(this.pulseTimer);
          this.pulseTimer = 0;
          let nextExpiry = Infinity;
          for (const event of this.pulseEvents.values()) {
            const expiresAt = event.firedAt + PULSE_SECONDS;
            if (expiresAt > now / 1000) nextExpiry = Math.min(nextExpiry, expiresAt);
          }
          if (Number.isFinite(nextExpiry)) {
            this.pulseTimer = window.setTimeout(() => { this.pulseTimer = 0; this.render(); }, nextExpiry * 1000 - now + 1);
          }
        } else if (!this.reducedMotion && (progress < 1 || now / 1000 < this.pulseUntil)) this.render();
      });
    }, Math.max(0, FRAME_MS-(performance.now()-this.lastRender)));
  };

  dispose(): void {
    this.disposed = true; this.cancelFrame();
    this.observer.disconnect(); this.intersection.disconnect(); this.controls.dispose();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.renderer.domElement.removeEventListener('keydown', this.onKey);
    this.pointGeometry.dispose(); this.haloGeometry.dispose(); this.lineGeometry.dispose(); this.pulseGeometry.dispose();
    this.pointMaterial.dispose(); this.haloMaterial.dispose(); this.lineMaterial.dispose(); this.pulseMaterial.dispose(); this.texture.dispose();
    this.pulseEvents.clear(); this.somaIndex.clear(); this.bodyIndex.clear();
    this.target.dispose(); this.compositeMaterial.dispose(); this.compositeGeometry.dispose(); this.compositeScene.clear();
    this.scene.clear(); this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
