import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const ACCEPTED_EXTS = [".ply", ".stl", ".obj", ".gltf", ".glb"];

// ---------------------------------------------------------------------------
// Helpers (defined outside the component — no closures over state)
// ---------------------------------------------------------------------------

function disposeObject(obj: THREE.Object3D) {
  obj.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const mats = Array.isArray(child.material)
        ? child.material
        : [child.material];
      mats.forEach((m: THREE.Material) => m.dispose());
    }
  });
}

function fitToView(object: THREE.Object3D) {
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim === 0) return;
  const scale = 4 / maxDim;
  // center at origin, then scale
  object.position.sub(center).multiplyScalar(scale);
  object.scale.setScalar(scale);
}

function defaultMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xadd8e6,
    side: THREE.DoubleSide,
  });
}

// ---------------------------------------------------------------------------
// Base64 → typed-array helpers
// ---------------------------------------------------------------------------

function decodeB64F32(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}

function decodeB64I32(b64: string): Int32Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Int32Array(buf.buffer);
}

// ---------------------------------------------------------------------------
// Viridis colormap (8 anchor colours, linear interpolation)
// ---------------------------------------------------------------------------

const VIRIDIS: [number, number, number][] = [
  [0.267, 0.005, 0.329],
  [0.275, 0.194, 0.494],
  [0.213, 0.359, 0.551],
  [0.152, 0.498, 0.558],
  [0.122, 0.624, 0.533],
  [0.290, 0.742, 0.441],
  [0.628, 0.823, 0.284],
  [0.993, 0.906, 0.144],
];

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const n = VIRIDIS.length - 1;
  const scaled = t * n;
  const i = Math.floor(scaled);
  const f = scaled - i;
  if (i >= n) return VIRIDIS[n];
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function applyFeatureColors(
  group: THREE.Group,
  hksFlat: Float32Array,
  nVerts: number,
  nFeatures: number,
  featureIdx: number,
) {
  let vmin = Infinity;
  let vmax = -Infinity;
  for (let i = 0; i < nVerts; i++) {
    const v = hksFlat[i * nFeatures + featureIdx];
    if (isFinite(v)) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  }
  const range = vmax > vmin ? vmax - vmin : 1;

  const colors = new Float32Array(nVerts * 3);
  for (let i = 0; i < nVerts; i++) {
    const v = hksFlat[i * nFeatures + featureIdx];
    const t = isFinite(v) ? (v - vmin) / range : 0.5;
    const [r, g, b] = viridis(t);
    colors[3 * i] = r;
    colors[3 * i + 1] = g;
    colors[3 * i + 2] = b;
  }

  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      if (child.material instanceof THREE.MeshStandardMaterial) {
        child.material.vertexColors = true;
        child.material.needsUpdate = true;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ThreeContext {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  meshGroup: THREE.Group;
}

type StatusKind = "idle" | "loading" | "loaded" | "error";

interface HksData {
  hks: Float32Array;
  nVerts: number;
  nFeatures: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MeshDemo({ apiUrl = "" }: { apiUrl?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const threeRef = useRef<ThreeContext | null>(null);
  const rafRef = useRef<number>(0);

  const [statusKind, setStatusKind] = useState<StatusKind>("idle");
  const [statusMsg, setStatusMsg] = useState(
    "Drop a mesh file here, or click Upload",
  );
  const [isDragging, setIsDragging] = useState(false);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [hksData, setHksData] = useState<HksData | null>(null);
  const [selectedFeature, setSelectedFeature] = useState(0);
  const [isComputing, setIsComputing] = useState(false);

  // -------------------------------------------------------------------------
  // Three.js initialisation
  // -------------------------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const w = container.clientWidth || 600;
    const h = container.clientHeight || 500;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x18181b); // zinc-900

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1e8);
    camera.position.set(0, 0, 6);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Lighting — key + fill
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(5, 10, 7);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.3);
    fill.position.set(-5, -5, -5);
    scene.add(fill);

    const meshGroup = new THREE.Group();
    scene.add(meshGroup);

    threeRef.current = { renderer, scene, camera, controls, meshGroup };

    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Keep canvas resolution in sync when the container resizes
    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      if (threeRef.current) {
        disposeObject(threeRef.current.meshGroup);
        threeRef.current.renderer.dispose();
        threeRef.current = null;
      }
    };
  }, []);

  // -------------------------------------------------------------------------
  // Mesh loading
  // -------------------------------------------------------------------------

  const loadMesh = useCallback(async (file: File) => {
    const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
    if (!ACCEPTED_EXTS.includes(ext)) {
      setStatusKind("error");
      setStatusMsg(
        `Unsupported format "${ext}". Accepted: ${ACCEPTED_EXTS.join(", ")}`,
      );
      return;
    }

    const ctx = threeRef.current;
    if (!ctx) return;

    setStatusKind("loading");
    setStatusMsg(`Loading ${file.name}…`);
    setHksData(null);
    setSelectedFeature(0);

    // Clear any previously loaded mesh
    while (ctx.meshGroup.children.length > 0) {
      const child = ctx.meshGroup.children[0];
      ctx.meshGroup.remove(child);
      disposeObject(child);
    }

    const url = URL.createObjectURL(file);
    try {
      let object: THREE.Object3D;

      if (ext === ".ply") {
        const geo = await new PLYLoader().loadAsync(url);
        geo.computeVertexNormals();
        object = new THREE.Mesh(geo, defaultMaterial());
      } else if (ext === ".stl") {
        const geo = await new STLLoader().loadAsync(url);
        geo.computeVertexNormals();
        object = new THREE.Mesh(geo, defaultMaterial());
      } else if (ext === ".obj") {
        object = await new OBJLoader().loadAsync(url);
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.material = defaultMaterial();
            child.geometry.computeVertexNormals();
          }
        });
      } else {
        // .gltf or .glb
        const gltf = await new GLTFLoader().loadAsync(url);
        object = gltf.scene;
      }

      fitToView(object);
      ctx.meshGroup.add(object);

      // Reset camera to a clean viewing position
      ctx.camera.position.set(0, 0, 6);
      ctx.camera.lookAt(0, 0, 0);
      ctx.controls.target.set(0, 0, 0);
      ctx.controls.update();

      // Tally geometry for the status line
      let nVerts = 0;
      let nFaces = 0;
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          const g = child.geometry as THREE.BufferGeometry;
          const pos = g.attributes.position;
          if (pos) nVerts += pos.count;
          if (g.index) nFaces += g.index.count / 3;
          else if (pos) nFaces += pos.count / 3;
        }
      });

      setCurrentFile(file);
      setStatusKind("loaded");
      setStatusMsg(
        `${file.name}  ·  ${nVerts.toLocaleString()} vertices, ${Math.round(nFaces).toLocaleString()} faces`,
      );
    } catch (err) {
      setStatusKind("error");
      setStatusMsg(`Failed to load mesh: ${(err as Error).message}`);
    } finally {
      URL.revokeObjectURL(url);
    }
  }, []);

  // -------------------------------------------------------------------------
  // Event handlers
  // -------------------------------------------------------------------------

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) loadMesh(file);
    },
    [loadMesh],
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) loadMesh(file);
      e.target.value = ""; // allow re-uploading the same file
    },
    [loadMesh],
  );

  const computeHks = useCallback(async () => {
    if (!currentFile || !apiUrl) return;
    const ctx = threeRef.current;
    if (!ctx) return;

    setIsComputing(true);
    setStatusKind("loading");
    setStatusMsg(`Sending ${currentFile.name} to backend…`);

    const formData = new FormData();
    formData.append("mesh_file", currentFile);

    try {
      const resp = await fetch(`${apiUrl}/compute-hks`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error((err as { detail?: string }).detail ?? resp.statusText);
      }

      const data = (await resp.json()) as {
        vertices: string;
        faces: string;
        hks: string;
        n_vertices: number;
        n_faces: number;
        n_features: number;
      };

      const vertices = decodeB64F32(data.vertices);
      const facesI32 = decodeB64I32(data.faces);
      const hks = decodeB64F32(data.hks);
      const nVerts = data.n_vertices;
      const nFaces = data.n_faces;
      const nFeatures = data.n_features;

      // Rebuild mesh from simplified geometry returned by the backend
      while (ctx.meshGroup.children.length > 0) {
        const child = ctx.meshGroup.children[0];
        ctx.meshGroup.remove(child);
        disposeObject(child);
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      geo.setIndex(
        new THREE.BufferAttribute(new Uint32Array(facesI32.buffer), 1),
      );
      geo.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({
        side: THREE.DoubleSide,
        vertexColors: true,
      });
      const mesh = new THREE.Mesh(geo, mat);
      fitToView(mesh);
      ctx.meshGroup.add(mesh);

      ctx.camera.position.set(0, 0, 6);
      ctx.camera.lookAt(0, 0, 0);
      ctx.controls.target.set(0, 0, 0);
      ctx.controls.update();

      const newHksData: HksData = { hks, nVerts, nFeatures };
      setHksData(newHksData);
      setSelectedFeature(0);
      applyFeatureColors(ctx.meshGroup, hks, nVerts, nFeatures, 0);

      setStatusKind("loaded");
      setStatusMsg(
        `HKS computed · ${nVerts.toLocaleString()} vertices (simplified) · ${nFeatures} features`,
      );
    } catch (err) {
      setStatusKind("error");
      setStatusMsg(`Compute HKS failed: ${(err as Error).message}`);
    } finally {
      setIsComputing(false);
    }
  }, [apiUrl, currentFile]);

  // Re-colour whenever the selected feature changes
  useEffect(() => {
    if (!hksData || !threeRef.current) return;
    applyFeatureColors(
      threeRef.current.meshGroup,
      hksData.hks,
      hksData.nVerts,
      hksData.nFeatures,
      selectedFeature,
    );
  }, [selectedFeature, hksData]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const msgColour =
    statusKind === "error"
      ? "text-red-400"
      : statusKind === "loaded"
        ? "text-zinc-400"
        : "text-zinc-500";

  return (
    <div className="not-prose flex flex-col gap-3">
      {/* 3-D viewer -------------------------------------------------------- */}
      <div
        ref={containerRef}
        className={`relative w-full overflow-hidden rounded-xl border-2 bg-zinc-900 transition-colors duration-150 ${
          isDragging ? "border-blue-400" : "border-zinc-700"
        }`}
        style={{ height: "500px" }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0 block h-full w-full"
        />

        {/* Idle drop-zone overlay */}
        {statusKind === "idle" && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2">
            <svg
              className="h-8 w-8 text-zinc-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
            <p className="text-sm text-zinc-500">Drop a mesh file here</p>
          </div>
        )}

        {/* Loading overlay */}
        {statusKind === "loading" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-zinc-900/70">
            <p className="text-sm text-zinc-300">Loading…</p>
          </div>
        )}
      </div>

      {/* Controls row ------------------------------------------------------ */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="cursor-pointer rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500">
          Upload mesh
          <input
            type="file"
            accept={ACCEPTED_EXTS.join(",")}
            className="hidden"
            onChange={handleFileInput}
          />
        </label>

        <span className={`truncate text-sm ${msgColour}`}>{statusMsg}</span>

        <button
          onClick={computeHks}
          disabled={!currentFile || isComputing || !apiUrl}
          title={
            !apiUrl
              ? "No backend configured (PUBLIC_HKS_API_URL not set)"
              : !currentFile
                ? "Upload a mesh first"
                : isComputing
                  ? "Computing…"
                  : "Run condensed_hks_pipeline on this mesh"
          }
          className={`ml-auto rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            currentFile && !isComputing && apiUrl
              ? "cursor-pointer bg-green-600 text-white hover:bg-green-500"
              : "cursor-not-allowed bg-zinc-700 text-zinc-500"
          }`}
        >
          {isComputing ? "Computing…" : "Compute HKS"}
          {!apiUrl && (
            <span className="ml-1.5 rounded bg-zinc-600 px-1.5 py-0.5 text-xs text-zinc-400">
              no backend
            </span>
          )}
        </button>
      </div>

      {/* HKS feature slider — visible after a successful compute ----------- */}
      {hksData && (
        <div className="flex items-center gap-3">
          <span className="whitespace-nowrap text-sm text-zinc-400">
            HKS timescale:{" "}
            <span className="font-mono text-zinc-200">{selectedFeature}</span>
            <span className="text-zinc-600"> / {hksData.nFeatures - 1}</span>
          </span>
          <input
            type="range"
            min={0}
            max={hksData.nFeatures - 1}
            step={1}
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(Number(e.target.value))}
            className="flex-1 accent-blue-400"
          />
        </div>
      )}
    </div>
  );
}
