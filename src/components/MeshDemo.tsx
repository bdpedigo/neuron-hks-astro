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

function decodeB64I8(b64: string): Int8Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Int8Array(buf.buffer);
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
// Label-keyed color palette for predictions
// ---------------------------------------------------------------------------

const DEFAULT_LABEL_COLORS: Record<string, string> = {
  soma: "#00beff",
  shaft: "#ddb310",
  not_spine: "#ddb310",
  spine: "#fb49b0",
  single_spine: "#fb49b0",
  multi_spine: "#9113cc",
  unknown: "#646464",
};

const FALLBACK_COLOR = "#646464";

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function applyPredictionColors(
  group: THREE.Group,
  labels: Int8Array,
  nVerts: number,
  classes: string[],
  colorMap: Record<string, string>,
) {
  const colors = new Float32Array(nVerts * 3);
  for (let i = 0; i < nVerts; i++) {
    const idx = labels[i];
    const label = idx >= 0 && idx < classes.length ? classes[idx] : "unknown";
    const [r, g, b] = hexToRgb(colorMap[label] ?? FALLBACK_COLOR);
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

// ---------------------------------------------------------------------------
// Wake up a sleeping HuggingFace Space by polling /health until it responds.
// HF Spaces go to sleep after 48 h of inactivity and take ~30–60 s to restart.
// ---------------------------------------------------------------------------
async function wakeBackend(
  apiUrl: string,
  onStatus: (msg: string) => void,
  maxWaitMs = 120_000,
  intervalMs = 5_000,
): Promise<void> {
  const deadline = Date.now() + maxWaitMs;
  let attempt = 0;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(`${apiUrl}/health`, { method: "GET" });
      if (resp.ok) return; // backend is awake
    } catch {
      // network error while cold-starting — keep polling
    }
    attempt++;
    const elapsed = Math.round((attempt * intervalMs) / 1000);
    onStatus(`Waking up backend… (${elapsed} s elapsed, please wait)`);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error("Backend did not wake up in time — please try again.");
}

interface HksData {
  hks: Float32Array;
  nVerts: number;
  nFeatures: number;
}

interface PredictionsData {
  labels: Int8Array;
  classes: string[];
  nVerts: number;
}

type ColorMode = "hks" | "predictions";

// ---------------------------------------------------------------------------
// Datastack presets
// ---------------------------------------------------------------------------

const DATASTACK_PRESETS = [
  {
    key: "microns",
    label: "MICrONS",
    cloudPath: "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
    rootId: "864691135307555142",
  },
  {
    key: "h01",
    label: "H01",
    cloudPath: "precomputed://gs://h01-release/data/20210601/c3",
    rootId: "664288036",
  },
];

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
    "Enter a cloud path and root ID, then click Compute HKS",
  );
  const [isDragging, setIsDragging] = useState(false);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [hksData, setHksData] = useState<HksData | null>(null);
  const [selectedFeature, setSelectedFeature] = useState(0);
  const [isComputing, setIsComputing] = useState(false);
  const [predictionsData, setPredictionsData] = useState<PredictionsData | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("hks");
  const [classColors, setClassColors] = useState<Record<string, string>>(DEFAULT_LABEL_COLORS);
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [inputMode, setInputMode] = useState<"upload" | "cloudvolume">("cloudvolume");
  const [cloudPath, setCloudPath] = useState("precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300");
  const [rootId, setRootId] = useState("864691135307555142");
  const [selectedPreset, setSelectedPreset] = useState("microns");

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

    const getViewerBg = () =>
      getComputedStyle(document.documentElement)
        .getPropertyValue("--viewer-bg")
        .trim() || "#d4d4d8";

    scene.background = new THREE.Color(getViewerBg());

    const themeObserver = new MutationObserver(() => {
      scene.background = new THREE.Color(getViewerBg());
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme", "class"],
    });

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
      themeObserver.disconnect();
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

  // -------------------------------------------------------------------------
  // Shared handler: decode a successful /compute-hks* response and update state
  // -------------------------------------------------------------------------

  const applyHksResponse = useCallback(
    (
      data: {
        vertices: string;
        faces: string;
        hks: string;
        n_vertices: number;
        n_faces: number;
        n_features: number;
        predictions?: string | null;
        classes?: string[] | null;
      },
      ctx: ThreeContext,
    ) => {
      const vertices = decodeB64F32(data.vertices);
      const facesI32 = decodeB64I32(data.faces);
      const hks = decodeB64F32(data.hks);
      const nVerts = data.n_vertices;
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

      if (data.predictions && data.classes) {
        setPredictionsData({
          labels: decodeB64I8(data.predictions),
          classes: data.classes,
          nVerts,
        });
        setClassColors((prev) => {
          const merged = { ...prev };
          for (const cls of data.classes!) {
            if (!(cls in merged)) {
              merged[cls] = DEFAULT_LABEL_COLORS[cls] ?? FALLBACK_COLOR;
            }
          }
          return merged;
        });
      } else {
        setPredictionsData(null);
      }

      setStatusKind("loaded");
      setStatusMsg(
        `HKS computed · ${nVerts.toLocaleString()} vertices (simplified) · ${nFeatures} features`,
      );
    },
    [],
  );

  const computeHks = useCallback(async () => {
    if (!currentFile || !apiUrl) return;
    const ctx = threeRef.current;
    if (!ctx) return;

    setIsComputing(true);
    setStatusKind("loading");
    setStatusMsg("Waking up backend…");
    setColorMode("hks");
    setPredictionsData(null);

    const formData = new FormData();
    formData.append("mesh_file", currentFile);

    try {
      await wakeBackend(apiUrl, setStatusMsg);
      setStatusMsg(`Sending ${currentFile.name} to backend…`);
      const resp = await fetch(`${apiUrl}/compute-hks?include_predictions=true`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error((err as { detail?: string }).detail ?? resp.statusText);
      }
      const data = await resp.json();
      applyHksResponse(data, ctx);
    } catch (err) {
      setStatusKind("error");
      setStatusMsg(`Compute HKS failed: ${(err as Error).message}`);
    } finally {
      setIsComputing(false);
    }
  }, [apiUrl, currentFile, applyHksResponse]);

  const computeHksFromCloud = useCallback(async () => {
    if (!cloudPath || !rootId || !apiUrl) return;
    const ctx = threeRef.current;
    if (!ctx) return;

    setIsComputing(true);
    setStatusKind("loading");
    setStatusMsg("Waking up backend…");
    setColorMode("hks");
    setPredictionsData(null);

    try {
      await wakeBackend(apiUrl, setStatusMsg);
      setStatusMsg(`Running pipeline… this may take a moment`);
      const resp = await fetch(`${apiUrl}/compute-hks-from-cloudvolume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cloud_path: cloudPath,
          root_id: rootId,
          include_predictions: true,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error((err as { detail?: string }).detail ?? resp.statusText);
      }
      const data = await resp.json();
      applyHksResponse(data, ctx);
    } catch (err) {
      setStatusKind("error");
      setStatusMsg(`Compute HKS failed: ${(err as Error).message}`);
    } finally {
      setIsComputing(false);
    }
  }, [apiUrl, cloudPath, rootId, applyHksResponse]);

  // Re-colour whenever the selected feature or color mode changes
  useEffect(() => {
    if (!threeRef.current) return;
    if (colorMode === "predictions" && predictionsData) {
      applyPredictionColors(
        threeRef.current.meshGroup,
        predictionsData.labels,
        predictionsData.nVerts,
        predictionsData.classes,
        classColors,
      );
    } else if (hksData) {
      applyFeatureColors(
        threeRef.current.meshGroup,
        hksData.hks,
        hksData.nVerts,
        hksData.nFeatures,
        selectedFeature,
      );
    }
  }, [colorMode, selectedFeature, hksData, predictionsData, classColors]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const msgColour =
    statusKind === "error"
      ? "text-red-400"
      : statusKind === "loaded"
        ? "text-zinc-400"
        : "text-zinc-500";

  const switchMode = (mode: "upload" | "cloudvolume") => {
    if (mode === inputMode) return;
    setInputMode(mode);
    setCurrentFile(null);
    setHksData(null);
    setPredictionsData(null);
    setStatusKind("idle");
    setStatusMsg(
      mode === "upload"
        ? "Drop a mesh file here, or click Upload"
        : "Enter a cloud path and root ID, then click Compute HKS",
    );
  };

  return (
    <div className="not-prose flex flex-col gap-3">
      {/* Mode toggle ------------------------------------------------------- */}
      <div className="flex overflow-hidden rounded-md border border-zinc-700 self-start">
        <button
          onClick={() => switchMode("cloudvolume")}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            inputMode === "cloudvolume"
              ? "bg-blue-600 text-white"
              : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
          }`}
        >
          Cloud path
        </button>
        <button
          onClick={() => switchMode("upload")}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            inputMode === "upload"
              ? "bg-blue-600 text-white"
              : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
          }`}
        >
          Upload mesh
        </button>
      </div>

      {/* 3-D viewer -------------------------------------------------------- */}
      <div
        ref={containerRef}
        className={`relative w-full overflow-hidden rounded-xl border-2 bg-zinc-900 transition-colors duration-150 ${
          isDragging ? "border-blue-400" : "border-zinc-700"
        }`}
        style={{ height: "500px" }}
        onDragOver={(e) => {
          if (inputMode !== "upload") return;
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={inputMode === "upload" ? handleDrop : undefined}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0 block h-full w-full"
        />

        {/* Idle drop-zone overlay */}
        {statusKind === "idle" && inputMode === "upload" && (
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
        {inputMode === "upload" ? (
          <label className="cursor-pointer rounded-md bg-zinc-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-600">
            Upload mesh
            <input
              type="file"
              accept={ACCEPTED_EXTS.join(",")}
              className="hidden"
              onChange={handleFileInput}
            />
          </label>
        ) : null}

        <span className={`truncate text-sm ${msgColour}`}>{statusMsg}</span>

        {inputMode === "upload" ? (
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
        ) : (
          <button
            onClick={computeHksFromCloud}
            disabled={!cloudPath || !rootId || isComputing || !apiUrl}
            title={
              !apiUrl
                ? "No backend configured (PUBLIC_HKS_API_URL not set)"
                : !cloudPath || !rootId
                  ? "Enter a cloud path and root ID first"
                  : isComputing
                    ? "Computing…"
                    : "Fetch mesh from CloudVolume and run condensed_hks_pipeline"
            }
            className={`ml-auto rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              cloudPath && rootId && !isComputing && apiUrl
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
        )}
      </div>

      {/* Cloud path inputs — shown below controls row in cloudvolume mode */}
      {inputMode === "cloudvolume" && (
        <div className="flex flex-col gap-3">
          <p className="text-sm text-zinc-400">
            The cloud path should be a publicly accessible cloud storage bucket that can be
            read by Neuroglancer (e.g.{" "}
            <span className="font-mono text-zinc-300">precomputed://gs://bucket/path</span>).
          </p>
          <div className="flex items-center gap-2">
            <label className="whitespace-nowrap text-sm text-zinc-400">Choose a public datastack:</label>
            <select
              value={selectedPreset}
              onChange={(e) => {
                const preset = DATASTACK_PRESETS.find((p) => p.key === e.target.value);
                if (preset) {
                  setSelectedPreset(preset.key);
                  setCloudPath(preset.cloudPath);
                  setRootId(preset.rootId);
                } else {
                  setSelectedPreset("");
                }
              }}
              className="min-w-[10rem] rounded-md border border-zinc-600 bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 focus:border-blue-500 focus:outline-none"
            >
              <option value="">Custom</option>
              {DATASTACK_PRESETS.map((p) => (
                <option key={p.key} value={p.key}>{p.label}</option>
              ))}
            </select>
          </div>
          <input
            type="text"
            placeholder="Cloud path (e.g. precomputed://gs://…)"
            value={cloudPath}
            onChange={(e) => { setCloudPath(e.target.value); setSelectedPreset(""); }}
            className="w-full rounded-md border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-blue-500 focus:outline-none"
          />
          <input
            type="text"
            inputMode="numeric"
            placeholder="Root ID"
            value={rootId}
            onChange={(e) => { setRootId(e.target.value.replace(/\D/g, "")); setSelectedPreset(""); }}
            className="w-full rounded-md border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-blue-500 focus:outline-none"
          />
        </div>
      )}

      {/* HKS feature slider — visible in HKS mode after a successful compute */}
      {hksData && colorMode === "hks" && (
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

      {/* Color mode toggle — visible once predictions are available -------- */}
      {predictionsData && (
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-zinc-400">Color by:</span>
          <div className="flex overflow-hidden rounded-md border border-zinc-700">
            <button
              onClick={() => setColorMode("hks")}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                colorMode === "hks"
                  ? "bg-blue-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              HKS timescale
            </button>
            <button
              onClick={() => setColorMode("predictions")}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                colorMode === "predictions"
                  ? "bg-orange-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              Synapse prediction
            </button>
          </div>
          {colorMode === "predictions" && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-zinc-400">
              {predictionsData.classes.map((cls) => (
                <span key={cls} className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-3 w-3 flex-shrink-0 rounded-sm"
                    style={{ backgroundColor: classColors[cls] ?? FALLBACK_COLOR }}
                  />
                  {cls}
                </span>
              ))}
              <button
                onClick={() => setShowColorPicker((v) => !v)}
                className="ml-1 rounded px-2 py-0.5 text-xs text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-zinc-300"
              >
                {showColorPicker ? "Hide colors" : "Customize colors"}
              </button>
            </div>
          )}
          {colorMode === "predictions" && showColorPicker && (
            <div className="flex flex-wrap gap-3 rounded-md border border-zinc-700 bg-zinc-800/50 p-3">
              {predictionsData.classes.map((cls) => (
                <label key={cls} className="flex cursor-pointer items-center gap-1.5 text-sm text-zinc-300">
                  <input
                    type="color"
                    value={classColors[cls] ?? FALLBACK_COLOR}
                    onChange={(e) =>
                      setClassColors((prev) => ({ ...prev, [cls]: e.target.value }))
                    }
                    className="h-6 w-6 cursor-pointer rounded border-0 bg-transparent p-0"
                  />
                  {cls}
                </label>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
