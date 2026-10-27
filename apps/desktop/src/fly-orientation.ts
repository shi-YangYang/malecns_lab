import * as THREE from 'three';

/** Illustrative orientation marker, not a reconstruction from MaleCNS data.
 * Model-local +Z is anterior and +Y dorsal; a rigid half-turn aligns these
 * with the brain display's world -Z (head) and -Y (dorsal view). */
export class FlyOrientation {
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-2.65, 2.65, 2.65, -2.65, .1, 20);
  private model = new THREE.Group();
  private viewport = new THREE.Vector4();
  private scissor = new THREE.Vector4();
  private geometries = new Set<THREE.BufferGeometry>();
  private materials = new Set<THREE.Material>();

  constructor(private anchor: HTMLElement) {
    const shell = new THREE.MeshStandardMaterial({ color: 0x9a927d, roughness: .58, metalness: .12 });
    const thorax = new THREE.MeshStandardMaterial({ color: 0x667c80, roughness: .58, metalness: .15 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x3b4d53, roughness: .65 });
    const eye = new THREE.MeshStandardMaterial({ color: 0xa4563e, roughness: .4, metalness: .08 });
    const wing = new THREE.MeshStandardMaterial({ color: 0xb6dce3, transparent: true, opacity: .22, roughness: .35, side: THREE.DoubleSide, depthWrite: false });
    const vein = new THREE.LineBasicMaterial({ color: 0x91b5c1, transparent: true, opacity: .58, depthWrite: false });
    const seam = new THREE.LineBasicMaterial({ color: 0x394e52, transparent: true, opacity: .7 });
    [shell, thorax, dark, eye, wing, vein, seam].forEach(material => this.materials.add(material));
    const sphere = this.keep(new THREE.SphereGeometry(1, 24, 16));
    const facets = this.keep(new THREE.IcosahedronGeometry(1, 2));
    const cylinder = this.keep(new THREE.CylinderGeometry(.65, 1, 1, 8));
    const ellipsoid = (position: number[], scale: number[], material: THREE.Material, faceted = false): void => {
      const mesh = new THREE.Mesh(faceted ? facets : sphere, material);
      mesh.position.fromArray(position); mesh.scale.fromArray(scale); this.model.add(mesh);
    };
    const rod = (from: number[], to: number[], radius: number, material = dark): void => {
      const start = new THREE.Vector3().fromArray(from), end = new THREE.Vector3().fromArray(to);
      const direction = end.clone().sub(start);
      const mesh = new THREE.Mesh(cylinder, material);
      mesh.position.copy(start).add(end).multiplyScalar(.5);
      mesh.scale.set(radius, direction.length(), radius);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      this.model.add(mesh);
    };
    const line = (points: THREE.Vector3[], material: THREE.LineBasicMaterial): void => {
      this.model.add(new THREE.Line(this.keep(new THREE.BufferGeometry().setFromPoints(points)), material));
    };

    // Compact thorax, separate head, paired compound eyes and short antennae.
    ellipsoid([0, 0, .2], [.42, .35, .53], thorax);
    ellipsoid([0, .015, .86], [.35, .28, .27], shell);
    ellipsoid([0, -.16, 1.05], [.11, .11, .16], dark);
    for (const side of [-1, 1]) {
      ellipsoid([side * .29, .055, .93], [.22, .28, .23], eye, true);
      ellipsoid([side * .1, .03, 1.12], [.055, .065, .09], shell);
      rod([side * .1, .06, 1.15], [side * .2, .18, 1.38], .018);
      rod([side * .19, .17, 1.35], [side * .31, .23, 1.46], .008);
      rod([side * .22, .19, 1.38], [side * .22, .29, 1.43], .006);

      // Three articulated legs on each side; no motor animation is implied.
      for (let leg = 0; leg < 3; leg++) {
        const hip = [side * .3, -.15, .48 - leg * .34];
        const knee = [side * (.7 + leg * .08), -.27, .77 - leg * .67];
        const ankle = [side * (1.03 + leg * .1), -.57, .95 - leg * .83];
        const tip = [side * (1.2 + leg * .1), -.6, 1.07 - leg * .89];
        rod(hip, knee, .035); rod(knee, ankle, .024); rod(ankle, tip, .013);
        ellipsoid(knee, [.036, .036, .036], shell);
      }

      // Wings spread backwards from the thorax. Shape coordinates are x/z,
      // mapped into the dorsal plane with a slight lift towards the tips.
      const shape = new THREE.Shape();
      shape.moveTo(.25, .26);
      shape.bezierCurveTo(.65, .12, 1.8, -.29, 1.98, -1.17);
      shape.bezierCurveTo(2.08, -1.65, 1.48, -1.8, 1.07, -1.31);
      shape.bezierCurveTo(.69, -.88, .42, -.29, .25, .26);
      const wingGeometry = this.keep(new THREE.ShapeGeometry(shape, 22));
      const position = wingGeometry.getAttribute('position');
      for (let i = 0; i < position.count; i++) {
        const x = position.getX(i), z = position.getY(i);
        position.setXYZ(i, side * x, .29 + (x - .25) * .07, z);
      }
      wingGeometry.computeVertexNormals();
      this.model.add(new THREE.Mesh(wingGeometry, wing));
      const wingPoint = (x: number, z: number): THREE.Vector3 => new THREE.Vector3(side * x, .295 + (x - .25) * .07, z);
      line(shape.getPoints(36).map(point => wingPoint(point.x, point.y)), vein);
      for (const [midX, midZ, tipX, tipZ] of [[.92, -.27, 1.98, -1.17], [.78, -.5, 1.77, -1.62], [.63, -.6, 1.27, -1.5]]) {
        const curve = new THREE.QuadraticBezierCurve3(wingPoint(.27, .2), wingPoint(midX, midZ), wingPoint(tipX, tipZ));
        line(curve.getPoints(14), vein);
      }
      line([wingPoint(.83, -.53), wingPoint(1.04, -.62), wingPoint(1.25, -.66)], vein);
      // Small haltere behind the wing root.
      rod([side * .32, .08, -.18], [side * .57, .2, -.32], .014, shell);
      ellipsoid([side * .57, .2, -.32], [.055, .055, .065], shell);
    }

    ellipsoid([0, -.01, -.82], [.36, .28, .77], shell);
    for (let segment = 0; segment < 5; segment++) {
      const z = -.36 - segment * .235;
      const radius = Math.sqrt(Math.max(0, 1 - ((z + .82) / .77) ** 2));
      const points = Array.from({ length: 49 }, (_, i) => {
        const angle = i * Math.PI / 24;
        return new THREE.Vector3(Math.cos(angle) * .365 * radius, -.01 + Math.sin(angle) * .285 * radius, z);
      });
      line(points, seam);
    }
    ellipsoid([0, -.015, -1.48], [.16, .15, .19], dark);
    // A restrained dorsal seam makes the body readable at the marker's size.
    line([new THREE.Vector3(0, .351, .56), new THREE.Vector3(0, .363, .18), new THREE.Vector3(0, .3, -.13)], seam);
    this.model.rotation.x = Math.PI;
    this.scene.add(this.model);
    // Camera-local backing stays fixed while the fly follows the brain view.
    // Draw inside the shared canvas so it covers the brain, never the marker.
    const cardShape = new THREE.Shape();
    const edge = 2.6, corner = .25;
    cardShape.moveTo(-edge + corner, -edge);
    cardShape.lineTo(edge - corner, -edge);
    cardShape.quadraticCurveTo(edge, -edge, edge, -edge + corner);
    cardShape.lineTo(edge, edge - corner);
    cardShape.quadraticCurveTo(edge, edge, edge - corner, edge);
    cardShape.lineTo(-edge + corner, edge);
    cardShape.quadraticCurveTo(-edge, edge, -edge, edge - corner);
    cardShape.lineTo(-edge, -edge + corner);
    cardShape.quadraticCurveTo(-edge, -edge, -edge + corner, -edge);
    const cardMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    this.materials.add(cardMaterial);
    const card = new THREE.Mesh(this.keep(new THREE.ShapeGeometry(cardShape)), cardMaterial);
    card.position.z = -12;
    this.camera.add(card);
    this.scene.add(new THREE.AmbientLight(0xb9d3dc, 1.5));
    const key = new THREE.DirectionalLight(0xffead2, 2.8);
    key.position.set(-3, 4, 5);
    const fill = new THREE.DirectionalLight(0x9acede, 1.6);
    fill.position.set(4, -2, 2);
    // Camera-mounted lighting keeps the tiny reference legible on every side.
    this.camera.add(key, fill); this.scene.add(this.camera);
  }

  private keep<T extends THREE.BufferGeometry>(geometry: T): T {
    this.geometries.add(geometry); return geometry;
  }

  render(renderer: THREE.WebGLRenderer, brainCamera: THREE.Camera): void {
    const bounds = this.anchor.getBoundingClientRect(), canvas = renderer.domElement.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    // A fixed orthographic scale isolates this marker from brain zoom/pan.
    // The shared camera orientation gives exactly the same view rotation.
    this.camera.quaternion.copy(brainCamera.quaternion);
    this.camera.position.set(0, 0, 8).applyQuaternion(brainCamera.quaternion);
    this.camera.updateMatrixWorld();
    renderer.getViewport(this.viewport); renderer.getScissor(this.scissor);
    const scissorTest = renderer.getScissorTest(), autoClear = renderer.autoClear;
    try {
      renderer.autoClear = false;
      renderer.setViewport(bounds.left - canvas.left, canvas.bottom - bounds.bottom, bounds.width, bounds.height);
      renderer.setScissor(bounds.left - canvas.left, canvas.bottom - bounds.bottom, bounds.width, bounds.height);
      renderer.setScissorTest(true);
      // Retain the completed brain composite; only depth in the inset resets.
      renderer.clearDepth();
      renderer.render(this.scene, this.camera);
    } finally {
      renderer.setViewport(this.viewport); renderer.setScissor(this.scissor);
      renderer.setScissorTest(scissorTest); renderer.autoClear = autoClear;
    }
  }

  dispose(): void {
    this.geometries.forEach(geometry => geometry.dispose());
    this.materials.forEach(material => material.dispose());
    this.geometries.clear(); this.materials.clear(); this.scene.clear();
  }
}
