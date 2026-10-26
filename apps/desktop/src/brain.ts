import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Atlas } from '../../atlas/src/types';
import type { TelemetryBatch, TelemetryGroup } from './types';

export interface BrainLegendEntry { id: string; label: string; color: string; count: number }

// Annotation families in this export, not inferred neuropil boundaries.
// Soma coordinates alone cannot identify MB/CX/AL or reconstruct axons.
const FAMILIES = [
  { id: 'T4', label: 'T4', color: '#49b6ff', match: /^T4[a-d]_/ },
  { id: 'LC4', label: 'LC4', color: '#4ae6dd', match: /^LC4_/ },
  { id: 'LPLC2', label: 'LPLC2', color: '#93f3e6', match: /^LPLC2_/ },
  { id: 'R16', label: 'R16', color: '#b58bff', match: /^R16_/ },
  { id: 'DN', label: 'DN', color: '#ffd58c', match: /^DN(?:g02|p01|p03)_/ },
];
const TRANSITION_SECONDS = .45;
const PULSE_SECONDS = 1.25;
const FRAME_MS = 1000 / 30;
const MAX_PULSE_POINTS = 32; // The telemetry manifest's sampled-neuron bound.

/** Real soma positions, annotation colors, group means and sampled spike events. */
export class BrainView {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(37, 1, 0.01, 1000);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private geometry = new THREE.BufferGeometry();
  private pulseGeometry = new THREE.BufferGeometry();
  private pulsePositions = new Float32Array(MAX_PULSE_POINTS * 3);
  private pulseTimes = new Float32Array(MAX_PULSE_POINTS);
  private pulseStrengths = new Float32Array(MAX_PULSE_POINTS);
  private activity: Float32Array;
  private previousActivity: Float32Array;
  private selected: Float32Array;
  private center = new THREE.Vector3();
  private radius = 1;
  private material: THREE.ShaderMaterial;
  private haloMaterial: THREE.ShaderMaterial;
  private pulseMaterial: THREE.ShaderMaterial;
  private frame = 0;
  private timer = 0;
  private lastRender = 0;
  private transitionStarted = 0;
  private animateUntil = 0;
  private lastBatch = '';
  private disposed = false;
  private inView = true;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  private pointByBody: Map<string, number>;
  private legend: BrainLegendEntry[] = [];
  private groupPoints: { id: string; indices: number[] }[] = [];
  private observer: ResizeObserver;
  private intersection: IntersectionObserver;
  private onKey: (event: KeyboardEvent) => void;
  private onVisibility = (): void => {
    if (document.hidden) this.cancelFrame();
    else this.render();
  };

  constructor(private container: HTMLElement, private atlas: Atlas) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
    const pixelRatio = Math.min(devicePixelRatio, 1.5);
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.dataset.testid = 'brain-canvas';
    this.renderer.domElement.setAttribute('aria-label', '真实胞体图谱，颜色区分已有注释群组。方向键旋转，加减键缩放，Home复位。');
    this.renderer.domElement.tabIndex = 0;
    container.append(this.renderer.domElement);
    this.pointByBody = new Map(atlas.body_ids.map((id, index) => [id, index]));
    this.activity = new Float32Array(atlas.positioned_count);
    this.previousActivity = new Float32Array(atlas.positioned_count);
    this.selected = new Float32Array(atlas.positioned_count);
    const colors = new Float32Array(atlas.positioned_count * 3);
    const annotated = new Float32Array(atlas.positioned_count);
    const cold = new THREE.Color('#578dbb');
    for (let index = 0; index < atlas.positioned_count; index++) cold.toArray(colors, index * 3);
    for (const family of FAMILIES) {
      const members = new Set<number>();
      for (const group of atlas.groups ?? []) {
        if (!family.match.test(group.name)) continue;
        for (const index of group.point_indices) {
          if (Number.isInteger(index) && index >= 0 && index < atlas.positioned_count) members.add(index);
        }
      }
      const color = new THREE.Color(family.color);
      for (const index of members) { color.toArray(colors, index * 3); annotated[index] = 1; }
      if (members.size) this.legend.push({ id: family.id, label: family.label, color: family.color, count: members.size });
    }
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(atlas.positions, 3));
    this.geometry.setAttribute('regionColor', new THREE.BufferAttribute(colors, 3));
    this.geometry.setAttribute('annotated', new THREE.BufferAttribute(annotated, 1));
    this.geometry.setAttribute('activity', new THREE.BufferAttribute(this.activity, 1).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setAttribute('previousActivity', new THREE.BufferAttribute(this.previousActivity, 1).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setAttribute('selected', new THREE.BufferAttribute(this.selected, 1).setUsage(THREE.DynamicDrawUsage));
    this.geometry.computeBoundingSphere();
    this.geometry.computeBoundingBox();
    this.geometry.boundingBox?.getCenter(this.center);
    this.radius = this.geometry.boundingSphere?.radius || 1;
    this.camera.near = Math.max(this.radius / 1000, .001);
    this.camera.far = this.radius * 100;
    const uniforms = {
      pixelRatio: { value: pixelRatio }, highlight: { value: 0 }, transition: { value: 1 },
      time: { value: 0 }, breath: { value: 0 }, radius: { value: this.radius },
      cameraDistance: { value: this.radius * 3 }, halo: { value: 0 },
    };
    const vertexShader = `
      attribute float activity; attribute float previousActivity;
      attribute float selected; attribute float annotated; attribute vec3 regionColor;
      uniform float pixelRatio; uniform float transition; uniform float radius;
      uniform float cameraDistance; uniform float halo;
      varying float vActivity; varying float vSelected; varying float vDepth;
      varying float vAnnotated; varying vec3 vColor;
      void main() {
        vActivity = mix(previousActivity, activity, smoothstep(0.0, 1.0, transition));
        vSelected = selected; vAnnotated = annotated; vColor = regionColor;
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        vDepth = clamp((cameraDistance + radius - (-view.z)) / (2.0 * radius), 0.0, 1.0);
        gl_Position = projectionMatrix * view;
        float focus = 0.88 + 0.35 * vDepth;
        float size = 1.25 + 0.55 * annotated + 2.8 * sqrt(vActivity) + selected * 0.8;
        gl_PointSize = mix(size, 5.0 + annotated * 3.0 + 14.0 * sqrt(vActivity), halo) * focus * pixelRatio;
      }`;
    const fragmentShader = `
      uniform float highlight; uniform float halo; uniform float breath;
      varying float vActivity; varying float vSelected; varying float vDepth;
      varying float vAnnotated; varying vec3 vColor;
      void main() {
        float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
        if (r > 1.0) discard;
        float energy = sqrt(vActivity);
        vec3 color = mix(vColor, vec3(0.67, 0.96, 1.0), energy * 0.56);
        color = mix(color, vec3(1.0, 0.88, 0.64), max(0.0, energy - 0.7) * 1.1);
        float focus = 0.45 + 0.55 * vDepth;
        float core = (0.19 + vAnnotated * 0.29 + energy * 0.42) * (1.0 - smoothstep(0.15, 1.0, r));
        float glow = (0.012 + vAnnotated * 0.013 + energy * 0.045) * exp(-r * r * 4.0);
        float emphasis = mix(1.0, 0.60 + 0.40 * vSelected, highlight);
        float alpha = mix(core, glow, halo) * focus * emphasis;
        // A bounded display envelope on measured group activity, not a new spike.
        alpha *= 1.0 + breath * energy * 0.10;
        gl_FragColor = vec4(color, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`;
    this.material = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    this.haloMaterial = new THREE.ShaderMaterial({
      uniforms: { ...uniforms, halo: { value: 1 } }, vertexShader, fragmentShader,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    // Both passes use actual somas; no invented surface or connectivity.
    this.scene.add(new THREE.Points(this.geometry, this.haloMaterial));
    this.scene.add(new THREE.Points(this.geometry, this.material));
    this.pulseGeometry.setAttribute('position', new THREE.BufferAttribute(this.pulsePositions, 3).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setAttribute('firedAt', new THREE.BufferAttribute(this.pulseTimes, 1).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setAttribute('strength', new THREE.BufferAttribute(this.pulseStrengths, 1).setUsage(THREE.DynamicDrawUsage));
    this.pulseGeometry.setDrawRange(0, 0);
    this.pulseMaterial = new THREE.ShaderMaterial({
      uniforms: { time: uniforms.time, pixelRatio: uniforms.pixelRatio, reduced: { value: this.reducedMotion ? 1 : 0 } },
      vertexShader: `
        attribute float firedAt; attribute float strength;
        uniform float time; uniform float pixelRatio; uniform float reduced;
        varying float vAge; varying float vStrength;
        void main() {
          vAge = (time - firedAt) / 1.25; vStrength = strength;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (12.0 + mix(36.0 * clamp(vAge, 0.0, 1.0), 0.0, reduced)) * pixelRatio;
        }`,
      fragmentShader: `
        varying float vAge; varying float vStrength;
        void main() {
          if (vAge < 0.0 || vAge > 1.0) discard;
          float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
          if (r > 1.0) discard;
          float envelope = pow(1.0 - vAge, 2.0);
          float core = exp(-r * r * 11.0);
          float ring = exp(-pow((r - 0.62) * 9.0, 2.0)) * 0.26;
          gl_FragColor = vec4(vec3(0.65, 0.96, 1.0), (core + ring) * envelope * vStrength);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const pulses = new THREE.Points(this.pulseGeometry, this.pulseMaterial);
    pulses.frustumCulled = false;
    this.scene.add(pulses);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.center);
    this.controls.enableDamping = false;
    this.controls.enablePan = true;
    this.controls.minDistance = this.radius * .25;
    this.controls.maxDistance = this.radius * 10;
    this.controls.addEventListener('change', this.render);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.reset();
    this.onKey = event => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', 'Home'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'Home') { this.reset(); return; }
      const offset = this.camera.position.clone().sub(this.controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      if (event.key === 'ArrowLeft') spherical.theta -= .1;
      if (event.key === 'ArrowRight') spherical.theta += .1;
      if (event.key === 'ArrowUp') spherical.phi = Math.max(.05, spherical.phi - .1);
      if (event.key === 'ArrowDown') spherical.phi = Math.min(Math.PI - .05, spherical.phi + .1);
      if (event.key === '+' || event.key === '=') spherical.radius *= .9;
      if (event.key === '-') spherical.radius *= 1.1;
      spherical.radius = THREE.MathUtils.clamp(spherical.radius, this.controls.minDistance, this.controls.maxDistance);
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
      this.controls.update();
    };
    this.renderer.domElement.addEventListener('keydown', this.onKey);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.intersection = new IntersectionObserver(entries => {
      this.inView = entries.some(entry => entry.isIntersecting);
      if (this.inView) this.render(); else this.cancelFrame();
    });
    this.intersection.observe(container);
    this.resize();
    this.render();
  }

  getLegend(): BrainLegendEntry[] { return this.legend.map(entry => ({ ...entry })); }

  private fitDistance(): number {
    const verticalSlope = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * .84;
    const horizontalSlope = verticalSlope * this.camera.aspect;
    const inverseRotation = this.camera.quaternion.clone().invert();
    const point = new THREE.Vector3();
    const positions = this.geometry.attributes.position;
    let distance = this.controls.minDistance;
    for (let index = 0; index < positions.count; index++) {
      point.fromBufferAttribute(positions, index).sub(this.controls.target).applyQuaternion(inverseRotation);
      distance = Math.max(distance, point.z + Math.max(Math.abs(point.x) / horizontalSlope,
        Math.abs(point.y) / verticalSlope, this.camera.near * 2));
    }
    return distance;
  }

  private resize(): void {
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    const previousFit = this.fitDistance();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.camera.position.copy(this.controls.target).add(offset.multiplyScalar(this.fitDistance() / previousFit));
    this.controls.update();
    this.render();
  }

  reset(): void {
    this.controls.target.copy(this.center);
    this.camera.position.copy(this.center).add(new THREE.Vector3(0, 0, 1));
    this.camera.lookAt(this.center);
    this.camera.position.z = this.center.z + this.fitDistance();
    this.controls.update();
  }

  setGroups(groups: TelemetryGroup[]): void {
    this.groupPoints = groups.map(group => ({ id: group.id, indices: group.body_ids.flatMap(id => {
      const index = this.pointByBody.get(id); return index === undefined ? [] : [index];
    }) }));
    this.lastBatch = '';
    this.pulseGeometry.setDrawRange(0, 0);
  }

  update(rates: number[] | null, groupId: string, maxRate: number, batch?: TelemetryBatch | null): void {
    const now = performance.now() / 1000;
    const t = Math.min(1, Math.max(0, (now - this.transitionStarted) / TRANSITION_SECONDS));
    const blend = t * t * (3 - 2 * t);
    for (let index = 0; index < this.activity.length; index++) {
      this.previousActivity[index] += (this.activity[index] - this.previousActivity[index]) * blend;
    }
    this.activity.fill(0); this.selected.fill(0);
    const scale = Number.isFinite(maxRate) && maxRate > 0 ? maxRate : 1;
    this.groupPoints.forEach((group, column) => {
      const rate = rates?.[column];
      const value = rate !== undefined && Number.isFinite(rate) ? THREE.MathUtils.clamp(rate / scale, 0, 1) : 0;
      for (const index of group.indices) {
        // This remains a group mean, never an inferred individual firing event.
        this.activity[index] = Math.max(this.activity[index], value);
        if (group.id === groupId) this.selected[index] = 1;
      }
    });
    for (const name of ['activity', 'previousActivity', 'selected']) this.geometry.attributes[name].needsUpdate = true;
    this.material.uniforms.highlight.value = groupId ? 1 : 0;
    this.transitionStarted = now;
    this.animateUntil = Math.max(this.animateUntil, now + TRANSITION_SECONDS);
    if (batch) this.updatePulses(batch, now);
    else if (batch === null || rates === null) {
      this.lastBatch = '';
      this.pulseGeometry.setDrawRange(0, 0);
    }
    this.render();
  }

  private updatePulses(batch: TelemetryBatch, now: number): void {
    const key = `${batch.run_id}:${batch.execution_id}:${batch.batch_sequence}`;
    if (key === this.lastBatch) return;
    this.lastBatch = key;
    const events = new Map<number, { step: number; count: number }>();
    for (const event of batch.spike_events) {
      const index = this.pointByBody.get(event.body_id);
      if (index === undefined) continue;
      const previous = events.get(index);
      events.set(index, { step: Math.min(previous?.step ?? event.step, event.step), count: (previous?.count ?? 0) + 1 });
    }
    // A quiet batch does not cut short the previous recorded event's decay.
    if (!events.size) return;
    let point = 0;
    for (const [index, event] of events) {
      if (point >= MAX_PULSE_POINTS) break;
      for (let axis = 0; axis < 3; axis++) this.pulsePositions[point * 3 + axis] = this.atlas.positions[index * 3 + axis];
      // Stretch recorded event ordering for visibility. This screen-space halo
      // stays at the actual soma; it is not measured axonal propagation.
      const order = THREE.MathUtils.clamp((event.step - batch.step_start) / Math.max(1, batch.step_end - batch.step_start), 0, 1);
      this.pulseTimes[point] = now + (this.reducedMotion ? 0 : order * .18);
      this.pulseStrengths[point] = Math.min(1, .55 + Math.log2(event.count + 1) * .12);
      point++;
    }
    for (const name of ['position', 'firedAt', 'strength']) this.pulseGeometry.attributes[name].needsUpdate = true;
    this.pulseGeometry.setDrawRange(0, point);
    if (point) this.animateUntil = now + PULSE_SECONDS + .18;
  }

  private cancelFrame(): void {
    cancelAnimationFrame(this.frame); window.clearTimeout(this.timer);
    this.frame = 0; this.timer = 0;
  }

  private render = (): void => {
    if (this.disposed || this.frame || this.timer || document.hidden || !this.inView) return;
    const delay = Math.max(0, FRAME_MS - (performance.now() - this.lastRender));
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      if (this.disposed || document.hidden || !this.inView) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        if (this.disposed || document.hidden || !this.inView) return;
        const now = performance.now() / 1000;
        this.material.uniforms.time.value = now;
        this.material.uniforms.transition.value = this.reducedMotion ? 1 : Math.min(1, (now - this.transitionStarted) / TRANSITION_SECONDS);
        this.material.uniforms.breath.value = this.reducedMotion ? 0 : Math.sin((now - this.transitionStarted) * Math.PI * 2) * Math.max(0, 1 - (now - this.transitionStarted) / 1.5);
        this.material.uniforms.cameraDistance.value = this.camera.position.distanceTo(this.center);
        this.renderer.render(this.scene, this.camera);
        this.lastRender = performance.now();
        // Idle is event driven; 30 fps is used only during a short display transition.
        if (now < this.animateUntil) this.render();
      });
    }, delay);
  };

  dispose(): void {
    this.disposed = true;
    this.cancelFrame();
    this.observer.disconnect(); this.intersection.disconnect(); this.controls.dispose();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.renderer.domElement.removeEventListener('keydown', this.onKey);
    this.geometry.dispose(); this.pulseGeometry.dispose();
    this.material.dispose(); this.haloMaterial.dispose(); this.pulseMaterial.dispose();
    this.scene.clear(); this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
