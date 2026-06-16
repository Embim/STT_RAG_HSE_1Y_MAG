import {
  Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild,
  inject, signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UpperCasePipe } from '@angular/common';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ApiService } from '../../core/api.service';
import { EmbeddingPoint, EmbeddingTopic } from '../../core/api.types';

interface Hover { x: number; y: number; title: string; snippet: string; }

@Component({
  selector: 'app-space',
  standalone: true,
  imports: [FormsModule, UpperCasePipe],
  templateUrl: './space.component.html',
  styleUrl: './space.component.css',
})
export class SpaceComponent implements AfterViewInit, OnDestroy {
  @ViewChild('host', { static: true }) host!: ElementRef<HTMLDivElement>;
  private api = inject(ApiService);
  private zone = inject(NgZone);

  loading = signal(true);
  error = signal<string | null>(null);
  topics = signal<EmbeddingTopic[]>([]);
  count = signal(0);
  reducer = signal('');
  hover = signal<Hover | null>(null);
  selected = signal<EmbeddingPoint | null>(null);
  query = '';
  searching = signal(false);
  resultCount = signal<number | null>(null);

  // three.js state
  private renderer?: THREE.WebGLRenderer;
  private scene?: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private controls?: OrbitControls;
  private points?: THREE.Points;
  private highlight?: THREE.Points;
  private marker?: THREE.Mesh;
  private raf = 0;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(-2, -2);
  private data: EmbeddingPoint[] = [];
  private idToIndex = new Map<string, number>();
  private resizeObs?: ResizeObserver;
  private spriteTex?: THREE.CanvasTexture;
  private onControlsStart = (): void => { if (this.controls) this.controls.autoRotate = false; };

  ngAfterViewInit(): void {
    this.initThree();
    this.load();
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObs?.disconnect();
    const dom = this.renderer?.domElement;
    if (dom) {
      dom.removeEventListener('pointermove', this.onPointerMove);
      dom.removeEventListener('pointerleave', this.onPointerLeave);
      dom.removeEventListener('click', this.onClick);
    }
    this.controls?.removeEventListener('start', this.onControlsStart);
    this.controls?.dispose();
    this.disposePoints();
    this.spriteTex?.dispose();           // общий texture освобождаем один раз
    this.renderer?.forceContextLoss();   // dispose() сам по себе НЕ отпускает WebGL-контекст
    this.renderer?.dispose();
    if (dom?.parentElement) dom.parentElement.removeChild(dom);
    this.renderer = undefined;
  }

  // ── data ──────────────────────────────────────────────────────────
  private load(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.embeddingMap().subscribe({
      next: (r) => {
        this.loading.set(false);
        this.topics.set(r.topics);
        this.count.set(r.count);
        this.reducer.set(r.reducer);
        this.data = r.points;
        if (!r.points.length) { this.error.set('Пока нет данных — добавьте лекции в разделе «Добавить».'); return; }
        this.buildCloud(r.points, r.topics);
      },
      error: (e) => {
        this.loading.set(false);
        this.error.set(e?.error?.detail ?? 'Не удалось построить карту (нужна векторная БД с данными).');
      },
    });
  }

  // ── three setup ───────────────────────────────────────────────────
  private initThree(): void {
    const el = this.host.nativeElement;
    const w = el.clientWidth || 800;
    const h = el.clientHeight || 600;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 4000);
    this.camera.position.set(0, 0, 150);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(0x000000, 0); // прозрачный — звёздный фон сайта виден сквозь
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.45;
    this.controls.minDistance = 20;
    this.controls.maxDistance = 600;

    this.raycaster.params.Points = { threshold: 1.6 };

    const dom = this.renderer.domElement;
    dom.addEventListener('pointermove', this.onPointerMove);
    dom.addEventListener('pointerleave', this.onPointerLeave);
    dom.addEventListener('click', this.onClick);
    // пользователь крутит сам → выключаем авто-вращение
    this.controls.addEventListener('start', this.onControlsStart);

    this.resizeObs = new ResizeObserver(() => this.onResize());
    this.resizeObs.observe(el);

    // render loop вне Angular zone — чтобы не дёргать change detection 60 раз/сек
    this.zone.runOutsideAngular(() => this.animate());
  }

  private animate = (): void => {
    this.raf = requestAnimationFrame(this.animate);
    this.controls?.update();
    if (this.marker) this.marker.rotation.y += 0.02;
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  };

  private onResize(): void {
    const el = this.host.nativeElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h || !this.renderer || !this.camera) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  // ── build point cloud ─────────────────────────────────────────────
  private buildCloud(pts: EmbeddingPoint[], topics: EmbeddingTopic[]): void {
    if (!this.scene) return;
    this.disposePoints();
    this.idToIndex.clear();

    const colorByCluster = new Map<number, THREE.Color>();
    for (const t of topics) colorByCluster.set(t.cluster, new THREE.Color(t.color));

    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      positions[i * 3] = p.x; positions[i * 3 + 1] = p.y; positions[i * 3 + 2] = p.z;
      const c = colorByCluster.get(p.cluster) ?? new THREE.Color('#8aa');
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
      this.idToIndex.set(p.id, i);
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 2.4,
      vertexColors: true,
      map: this.sprite(),
      transparent: true,
      alphaTest: 0.35,
      sizeAttenuation: true,
      depthWrite: false,
    });
    this.points = new THREE.Points(geo, mat);
    this.scene.add(this.points);
  }

  /** Мягкая круглая «звёздочка» для точки. Создаётся ОДИН раз и переиспользуется
   *  всеми материалами (Material.dispose() не освобождает map → иначе утечка). */
  private sprite(): THREE.CanvasTexture {
    if (this.spriteTex) return this.spriteTex;
    const s = 64;
    const cv = document.createElement('canvas');
    cv.width = cv.height = s;
    const ctx = cv.getContext('2d')!;
    const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.4, 'rgba(255,255,255,0.85)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, s, s);
    const tex = new THREE.CanvasTexture(cv);
    tex.needsUpdate = true;
    this.spriteTex = tex;
    return tex;
  }

  private disposePoints(): void {
    for (const o of [this.points, this.highlight] as (THREE.Points | undefined)[]) {
      if (o) {
        this.scene?.remove(o);
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
    }
    this.points = this.highlight = undefined;
    if (this.marker) { this.scene?.remove(this.marker); this.marker.geometry.dispose(); (this.marker.material as THREE.Material).dispose(); this.marker = undefined; }
  }

  // ── hover / pick ──────────────────────────────────────────────────
  private onPointerMove = (e: PointerEvent): void => {
    const rect = this.renderer!.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const hit = this.pick();
    if (hit == null) {
      if (this.hover()) this.zone.run(() => this.hover.set(null));
      return;
    }
    const p = this.data[hit];
    this.zone.run(() => this.hover.set({
      x: e.clientX - rect.left, y: e.clientY - rect.top,
      title: p.title, snippet: p.snippet,
    }));
  };

  private onPointerLeave = (): void => { if (this.hover()) this.zone.run(() => this.hover.set(null)); };

  private onClick = (): void => {
    const hit = this.pick();
    if (hit == null) return;
    this.zone.run(() => this.selected.set(this.data[hit]));
  };

  private pick(): number | null {
    if (!this.points || !this.camera) return null;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.points, false);
    return hits.length ? (hits[0].index ?? null) : null;
  }

  // ── search → highlight + marker ───────────────────────────────────
  search(): void {
    const q = this.query.trim();
    if (!q || this.searching()) return;
    this.searching.set(true);
    this.resultCount.set(null);
    this.api.locateInMap(q, 8).subscribe({
      next: (r) => {
        this.searching.set(false);
        const shown = this.applyHighlight(r.highlight_ids);
        this.resultCount.set(shown);   // честно: сколько реально подсвечено в карте
        this.placeMarker(r.marker);
      },
      error: (e) => { this.searching.set(false); this.error.set(e?.error?.detail ?? 'Не удалось спроецировать запрос.'); },
    });
  }

  clearSearch(): void {
    this.query = '';
    this.resultCount.set(null);
    this.applyHighlight([]);
    this.placeMarker(null);
  }

  /** Подсветить точки по id. Возвращает, сколько РЕАЛЬНО найдено в текущем
   *  облаке (id из near_text может не попасть в карту при капе точек). */
  private applyHighlight(ids: string[]): number {
    if (!this.scene || !this.points) return 0;
    if (this.highlight) { this.scene.remove(this.highlight); this.highlight.geometry.dispose(); (this.highlight.material as THREE.Material).dispose(); this.highlight = undefined; }
    const idx = ids.map((id) => this.idToIndex.get(id)).filter((i): i is number => i != null);
    if (!idx.length) return 0;
    const src = (this.points.geometry.getAttribute('position') as THREE.BufferAttribute);
    const pos = new Float32Array(idx.length * 3);
    idx.forEach((i, k) => { pos[k * 3] = src.getX(i); pos[k * 3 + 1] = src.getY(i); pos[k * 3 + 2] = src.getZ(i); });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      size: 7, color: 0xffffff, map: this.sprite(),
      transparent: true, alphaTest: 0.2, sizeAttenuation: true, depthWrite: false,
    });
    this.highlight = new THREE.Points(geo, mat);
    this.scene.add(this.highlight);
    return idx.length;
  }

  private placeMarker(m: { x: number; y: number; z: number } | null): void {
    if (!this.scene) return;
    if (this.marker) { this.scene.remove(this.marker); this.marker.geometry.dispose(); (this.marker.material as THREE.Material).dispose(); this.marker = undefined; }
    if (!m) return;
    const geo = new THREE.OctahedronGeometry(3.4, 0);
    const mat = new THREE.MeshBasicMaterial({ color: 0xe3b341, wireframe: true });
    this.marker = new THREE.Mesh(geo, mat);
    this.marker.position.set(m.x, m.y, m.z);
    this.scene.add(this.marker);
    // плавно навести камеру на маркер
    if (this.controls) { this.controls.target.set(m.x, m.y, m.z); this.controls.autoRotate = false; }
  }

  // ── topic legend interaction ──────────────────────────────────────
  focusTopic(t: EmbeddingTopic): void {
    const ids = this.data.filter((p) => p.cluster === t.cluster).map((p) => p.id);
    this.resultCount.set(this.applyHighlight(ids));
    this.placeMarker(null);
  }

  openSelected(): string | null {
    const p = this.selected();
    if (!p?.source_url) return null;
    const t = p.start != null ? Math.floor(p.start) : 0;
    try {
      const u = new URL(p.source_url);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
      // убираем уже стоящие тайм-параметры, чтобы не конфликтовали с нашим t
      ['t', 'start', 'time_continue'].forEach((k) => u.searchParams.delete(k));
      if (t > 0) u.searchParams.set('t', `${t}s`);
      return u.toString();
    } catch {
      return null;
    }
  }

  fmtTime(sec: number | null): string {
    if (sec == null) return '';
    const s = Math.floor(sec);
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, '0')}`;
  }
}
