import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Atlas } from '../../atlas/src/types';
import type { TelemetryBatch, TelemetryGroup } from './types';
import type { BrainClip, BrainDisplayData } from './brain-data';

export interface BrainLegendEntry { id: string; label: string; color: string; count: number }
const FRAME_MS = 1000 / 30;
const TRANSITION_MS = 650;
const TEXTURE_WIDTH = 512;

/** Original synapse positions and measured scalar overlays. Straight graph
 * edges join representative synapses, never reconstructed axon trajectories. */
export class BrainView {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(34, 1, .01, 100);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private pointGeometry = new THREE.BufferGeometry();
  private lineGeometry = new THREE.BufferGeometry();
  private pointMaterial: THREE.ShaderMaterial;
  private lineMaterial: THREE.ShaderMaterial;
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
  private frame = 0;
  private timer = 0;
  private lastRender = 0;
  private transitionStart = 0;
  private inView = true;
  private disposed = false;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  private uniforms: { [name: string]: THREE.IUniform };
  private onVisibility = (): void => { if (document.hidden) this.cancelFrame(); else this.render(); };

  constructor(private container: HTMLElement, _atlas: Atlas | null, private data: BrainDisplayData) {
    const { positions, owners, centers, manifest: m } = data;
    this.bodyIndex = new Map(m.body_ids.map((id, i) => [id, i]));
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
          gl_PointSize = (4.5 + energy * 2.0) * pixelRatio * viewportScale;
        }`,
      fragmentShader: `
        varying float energy; varying float depth;
        void main() {
          vec2 p = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(p, p);
          if (r2 > 1.0) discard;
          float soft = exp(-4.5 * r2) - exp(-4.5);
          vec3 base = vec3(0.18, 0.39, 0.49);
          vec3 warm = vec3(1.0, 0.78, 0.43);
          vec3 color = mix(base, warm, smoothstep(0.02, 0.65, energy));
          float alpha = soft * (0.021 + energy * 0.24) * (0.30 + 0.70 * depth);
          gl_FragColor = vec4(color, alpha);
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
          gl_FragColor = vec4(mix(vec3(0.28, 0.48, 0.58), vec3(1.0, 0.80, 0.48), energy), 0.065 + 0.20 * energy);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.scene.add(new THREE.Points(this.pointGeometry, this.pointMaterial));
    this.scene.add(new THREE.LineSegments(this.lineGeometry, this.lineMaterial));
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
  }

  /** Workbench telemetry remains group-mean activity, never per-neuron spikes. */
  update(rates: number[] | null, groupId: string, maxRate: number, _batch?: TelemetryBatch | null): void {
    const values = new Float32Array(this.data.manifest.node_count);
    this.groups.forEach((g, c) => {
      const rate = rates?.[c] ?? 0;
      const value = Number.isFinite(rate) ? Math.max(0, Math.min(1, rate / Math.max(1, maxRate))) : 0;
      for (const i of g.indices) values[i] = Math.max(values[i], value * (groupId && groupId !== g.id ? .2 : 1));
    });
    this.setSignal(values);
  }

  /** Fixed common scale and actual L1/L2 order; action names do not affect color. */
  setRecordedFrame(clip: BrainClip | null, half = 0): void {
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
    this.setSignal(values);
  }

  private setSignal(values: Float32Array): void {
    const progress = THREE.MathUtils.clamp((performance.now()-this.transitionStart)/TRANSITION_MS, 0, 1);
    const blend = progress * progress * (3-2*progress);
    for (let i = 0; i < values.length; i++) {
      this.signal[i*4] += (this.signal[i*4+1]-this.signal[i*4]) * blend;
      this.signal[i*4+1] = values[i];
    }
    this.texture.needsUpdate = true;
    this.transitionStart = performance.now();
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

  private cancelFrame(): void { cancelAnimationFrame(this.frame); clearTimeout(this.timer); this.frame = 0; this.timer = 0; }
  private render = (): void => {
    if (this.disposed || this.frame || this.timer || document.hidden || !this.inView) return;
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      if (this.disposed || document.hidden || !this.inView) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        if (this.disposed || document.hidden || !this.inView) return;
        const progress = Math.min(1, (performance.now()-this.transitionStart)/TRANSITION_MS);
        this.uniforms.blend.value = this.reducedMotion ? 1 : progress;
        this.uniforms.cameraDistance.value = this.camera.position.distanceTo(this.center);
        this.renderer.setRenderTarget(this.target);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.compositeScene, this.compositeCamera);
        this.lastRender = performance.now();
        if (progress < 1 && !this.reducedMotion) this.render();
      });
    }, Math.max(0, FRAME_MS-(performance.now()-this.lastRender)));
  };

  dispose(): void {
    this.disposed = true; this.cancelFrame();
    this.observer.disconnect(); this.intersection.disconnect(); this.controls.dispose();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.renderer.domElement.removeEventListener('keydown', this.onKey);
    this.pointGeometry.dispose(); this.lineGeometry.dispose(); this.pointMaterial.dispose(); this.lineMaterial.dispose(); this.texture.dispose();
    this.target.dispose(); this.compositeMaterial.dispose(); this.compositeGeometry.dispose(); this.compositeScene.clear();
    this.scene.clear(); this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
