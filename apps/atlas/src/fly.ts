import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/** A static, illustrative body. No recorded motor or pose data is available. */
export class FlyView {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  private controls: OrbitControls;
  private observer: ResizeObserver;
  private frame = 0;
  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.dataset.testid = 'fly-canvas';
    this.renderer.domElement.setAttribute('aria-label', '静态果蝇结构示意，可拖动旋转；没有神经驱动动作。');
    container.append(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xc9eee5, 0x373024, 2));
    const key = new THREE.DirectionalLight(0xffddaa, 3);
    key.position.set(3, 5, 2); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x7bb8be, 2);
    fill.position.set(-4, 1, -3); this.scene.add(fill);
    const shell = new THREE.MeshStandardMaterial({ color: 0x8c7960, roughness: 0.65, metalness: 0.15 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x343c37, roughness: 0.75 });
    const eye = new THREE.MeshStandardMaterial({ color: 0xa55d3c, roughness: 0.55, flatShading: true });
    const wing = new THREE.MeshPhysicalMaterial({ color: 0xb6d4c9, transparent: true, opacity: 0.24, roughness: 0.25, side: THREE.DoubleSide, depthWrite: false });
    const ellipsoid = (x: number, y: number, z: number, sx: number, sy: number, sz: number, material: THREE.Material, facets = false): THREE.Mesh => {
      const mesh = new THREE.Mesh(facets ? new THREE.IcosahedronGeometry(1, 2) : new THREE.SphereGeometry(1, 24, 16), material);
      mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz); this.scene.add(mesh); return mesh;
    };
    const rod = (start: THREE.Vector3, end: THREE.Vector3, radius: number, material = dark): void => {
      const direction = end.clone().sub(start);
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.65, radius, direction.length(), 7), material);
      mesh.position.copy(start).add(end).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      this.scene.add(mesh);
    };
    ellipsoid(0, 0, 0.1, 0.43, 0.4, 0.64, shell);
    ellipsoid(0, 0.06, 0.83, 0.41, 0.34, 0.32, dark);
    for (const side of [-1, 1]) {
      ellipsoid(side * 0.31, 0.1, 0.94, 0.22, 0.29, 0.22, eye, true);
      rod(new THREE.Vector3(side * 0.11, 0.15, 1.06), new THREE.Vector3(side * 0.2, 0.3, 1.38), 0.018);
      ellipsoid(side * 0.19, 0.29, 1.32, 0.04, 0.06, 0.08, shell);
      for (let leg = 0; leg < 3; leg++) {
        const start = new THREE.Vector3(side * 0.27, -0.1, 0.44 - leg * 0.45);
        const knee = new THREE.Vector3(side * (0.76 + leg * 0.12), -0.23, 0.9 - leg * 0.88);
        const ankle = new THREE.Vector3(side * (1.05 + leg * 0.19), -0.65, 1.1 - leg * 1.08);
        const tip = new THREE.Vector3(side * (1.23 + leg * 0.18), -0.73, 1.22 - leg * 1.18);
        rod(start, knee, 0.035); rod(knee, ankle, 0.025); rod(ankle, tip, 0.015);
      }
      const shape = new THREE.Shape();
      shape.moveTo(0.1, 0.2); shape.bezierCurveTo(0.55, 0.3, 2.2, -0.25, 2.1, -1.15);
      shape.bezierCurveTo(1.9, -1.8, 0.9, -1.2, 0.1, 0.2);
      const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape, 24), wing);
      mesh.rotation.x = -Math.PI / 2; mesh.scale.x = side; mesh.position.y = 0.24; this.scene.add(mesh);
      for (const tip of [[1.85, -1.15], [1.9, -0.7], [1.25, -1.12]]) {
        const points = [new THREE.Vector3(side * 0.15, 0.245, -0.2), new THREE.Vector3(side * tip[0], 0.245, -tip[1])];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        this.scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x9bb0a3, transparent: true, opacity: 0.28 })));
      }
    }
    for (let segment = 0; segment < 5; segment++) {
      const size = 0.4 * (1 - segment * 0.13);
      ellipsoid(0, -0.04, -0.58 - segment * 0.23, size, size * 0.76, 0.26, segment % 2 ? dark : shell);
    }
    this.camera.position.set(3.2, 3, 5.3);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -0.05, 0); this.controls.enableDamping = true;
    this.controls.enablePan = false; this.controls.enableZoom = false; this.controls.update();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(container);
    this.resize(); this.render();
  }
  private resize(): void {
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix(); this.renderer.setSize(width, height);
  }
  private render = (): void => {
    this.controls.update(); this.renderer.render(this.scene, this.camera); this.frame = requestAnimationFrame(this.render);
  };
  dispose(): void {
    cancelAnimationFrame(this.frame); this.observer.disconnect(); this.controls.dispose();
    this.scene.traverse(object => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material]; materials.forEach(material => material.dispose());
      }
    });
    this.renderer.dispose();
  }
}
