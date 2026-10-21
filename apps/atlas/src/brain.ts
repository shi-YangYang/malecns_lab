import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Atlas, Run } from './types';

export class BrainView {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(37, 1, 0.01, 1000);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private geometry = new THREE.BufferGeometry();
  private activity: Float32Array;
  private selected: Float32Array;
  private center = new THREE.Vector3();
  private material: THREE.ShaderMaterial;
  private frame = 0;
  private observer: ResizeObserver;
  private onKey: (event: KeyboardEvent) => void;

  constructor(private container: HTMLElement, private atlas: Atlas) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.dataset.testid = 'brain-canvas';
    this.renderer.domElement.setAttribute('aria-label', '真实胞体位置三维点云。方向键旋转，加减键缩放，Home 复位。');
    this.renderer.domElement.tabIndex = 0;
    container.append(this.renderer.domElement);
    this.activity = new Float32Array(atlas.positioned_count);
    this.selected = new Float32Array(atlas.positioned_count);
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(atlas.positions, 3));
    this.geometry.setAttribute('activity', new THREE.BufferAttribute(this.activity, 1).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setAttribute('selected', new THREE.BufferAttribute(this.selected, 1).setUsage(THREE.DynamicDrawUsage));
    this.geometry.computeBoundingSphere();
    this.geometry.computeBoundingBox();
    this.geometry.boundingBox?.getCenter(this.center);
    const radius = this.geometry.boundingSphere?.radius || 1;
    this.camera.near = Math.max(radius / 1000, 0.001);
    this.camera.far = radius * 100;
    this.material = new THREE.ShaderMaterial({
      uniforms: { pixelRatio: { value: Math.min(devicePixelRatio, 2) }, highlight: { value: 0 } },
      vertexShader: `
        attribute float activity; attribute float selected;
        varying float vActivity; varying float vSelected;
        uniform float pixelRatio;
        void main() {
          vActivity = activity; vSelected = selected;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (1.25 + 3.0 * sqrt(activity) + selected * 0.8) * pixelRatio;
        }`,
      fragmentShader: `
        varying float vActivity; varying float vSelected; uniform float highlight;
        void main() {
          float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
          if (r > 1.0) discard;
          vec3 cold = vec3(0.20, 0.40, 0.43);
          vec3 mint = vec3(0.33, 0.89, 0.76);
          vec3 amber = vec3(1.0, 0.73, 0.37);
          vec3 color = vActivity < 0.5 ? mix(cold, mint, vActivity * 2.0) : mix(mint, amber, (vActivity - 0.5) * 2.0);
          float alpha = (0.32 + 0.60 * sqrt(vActivity)) * (1.0 - smoothstep(0.3, 1.0, r));
          float emphasis = mix(1.0, 0.52 + 0.48 * vSelected, highlight);
          color += vSelected * highlight * vec3(0.10, 0.14, 0.13);
          gl_FragColor = vec4(color, alpha * emphasis);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.scene.add(new THREE.Points(this.geometry, this.material));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.center);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = true;
    this.controls.minDistance = radius * 0.25;
    this.controls.maxDistance = radius * 10;
    this.reset();
    this.onKey = event => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', 'Home'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'Home') { this.reset(); return; }
      const offset = this.camera.position.clone().sub(this.controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      if (event.key === 'ArrowLeft') spherical.theta -= 0.1;
      if (event.key === 'ArrowRight') spherical.theta += 0.1;
      if (event.key === 'ArrowUp') spherical.phi = Math.max(0.05, spherical.phi - 0.1);
      if (event.key === 'ArrowDown') spherical.phi = Math.min(Math.PI - 0.05, spherical.phi + 0.1);
      if (event.key === '+' || event.key === '=') spherical.radius *= 0.9;
      if (event.key === '-') spherical.radius *= 1.1;
      spherical.radius = THREE.MathUtils.clamp(spherical.radius, this.controls.minDistance, this.controls.maxDistance);
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
      this.controls.update();
    };
    this.renderer.domElement.addEventListener('keydown', this.onKey);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.render();
  }
  private fitDistance(): number {
    // Fit every real position to the perspective frustum, with an 8% edge margin.
    const verticalSlope = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 0.84;
    const horizontalSlope = verticalSlope * this.camera.aspect;
    const inverseRotation = this.camera.quaternion.clone().invert();
    const point = new THREE.Vector3();
    const positions = this.geometry.attributes.position;
    let distance = this.controls.minDistance;
    for (let index = 0; index < positions.count; index++) {
      point.fromBufferAttribute(positions, index).sub(this.controls.target).applyQuaternion(inverseRotation);
      distance = Math.max(distance, point.z + Math.max(
        Math.abs(point.x) / horizontalSlope,
        Math.abs(point.y) / verticalSlope,
        this.camera.near * 2,
      ));
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
    // Retain the orbit direction and the user's zoom relative to a fitted view.
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.camera.position.copy(this.controls.target).add(offset.multiplyScalar(this.fitDistance() / previousFit));
    this.controls.update();
  }
  reset(): void {
    this.controls.target.copy(this.center);
    this.camera.position.copy(this.center).add(new THREE.Vector3(0, 0, 1));
    this.camera.lookAt(this.center);
    this.camera.position.z = this.center.z + this.fitDistance();
    this.controls.update();
  }
  update(run: Run, frame: number, group: string, maxRate: number, highlight: boolean): void {
    this.activity.fill(0); this.selected.fill(0);
    const valid = run.valid[frame] === true;
    for (const item of this.atlas.groups) {
      const column = run.groups.indexOf(item.name);
      const value = valid && column >= 0 ? run.group_rates_hz[frame]?.[column] : null;
      for (const index of item.point_indices) {
        // A point in overlapping groups uses the largest recorded group mean.
        if (value !== null && value !== undefined) this.activity[index] = Math.max(this.activity[index], Math.min(value / maxRate, 1));
        if (item.name === group) this.selected[index] = 1;
      }
    }
    this.geometry.attributes.activity.needsUpdate = true;
    this.geometry.attributes.selected.needsUpdate = true;
    this.material.uniforms.highlight.value = highlight ? 1 : 0;
  }
  private render = (): void => {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.frame = requestAnimationFrame(this.render);
  };
  dispose(): void {
    cancelAnimationFrame(this.frame); this.observer.disconnect(); this.controls.dispose();
    this.renderer.domElement.removeEventListener('keydown', this.onKey);
    this.geometry.dispose(); this.material.dispose(); this.renderer.dispose();
  }
}
