import {
  Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild, inject,
} from '@angular/core';

interface Star    { x: number; y: number; z: number; r: number; a: number; tw: number; ph: number; c: string; }
interface Shoot   { x: number; y: number; vx: number; vy: number; life: number; max: number; len: number; rot: number; rspd: number; size: number; }
interface Orbiter { rr: number; ang: number; spd: number; size: number; c: number[]; rot: number; rspd: number; inc: number; squash: number; tphase: number; }
interface Gem     { x: number; y: number; vx: number; vy: number; R: number; rot: number; rspd: number; depth: number; base: number[]; }

@Component({
  selector: 'app-starfield',
  standalone: true,
  template: '<canvas #cv></canvas>',
  styles: [`
    :host { position: fixed; inset: 0; z-index: -2; pointer-events: none; display: block; }
    canvas { width: 100%; height: 100%; display: block; }
  `],
})
export class StarfieldComponent implements AfterViewInit, OnDestroy {
  @ViewChild('cv', { static: true }) cv!: ElementRef<HTMLCanvasElement>;
  private zone   = inject(NgZone);
  private ctx!:   CanvasRenderingContext2D;
  private raf    = 0;
  private w      = 0; private h = 0; private dpr = 1;
  private stars:   Star[]    = [];
  private shoots:  Shoot[]   = [];
  private orbiters: Orbiter[] = [];
  private gems:    Gem[]     = [];

  // Orbit center / scale — read from .bg-contour bounding box in resize()
  private ocx    = 0; private ocy = 0; private oscale = 0;

  // Global precession angle for tilted orbits
  private tilt = 0;

  private nextShoot = 2000;
  private last      = 0;
  private reduced   = false;
  private onResize  = () => this.resize();

  // Cursor parallax state
  private px = 0; private py = 0;
  private tpx = 0; private tpy = 0;
  private onPointerMove = (e: PointerEvent) => {
    this.tpx = (e.clientX / window.innerWidth  - 0.5) * 2;
    this.tpy = (e.clientY / window.innerHeight - 0.5) * 2;
  };

  ngAfterViewInit(): void {
    this.ctx     = this.cv.nativeElement.getContext('2d')!;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.resize();
    window.addEventListener('resize', this.onResize);
    if (this.reduced) {
      document.documentElement.style.setProperty('--par-x', '0');
      document.documentElement.style.setProperty('--par-y', '0');
      this.draw(0, true);
      return;
    }
    window.addEventListener('pointermove', this.onPointerMove);
    this.zone.runOutsideAngular(() => {
      const loop = (t: number) => { this.draw(t, false); this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('pointermove', this.onPointerMove);
  }

  private resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w   = window.innerWidth;
    this.h   = window.innerHeight;
    const c  = this.cv.nativeElement;
    c.width  = Math.floor(this.w * this.dpr);
    c.height = Math.floor(this.h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    // ── drifting star field ──────────────────────────────────────────────────
    const count   = Math.round(Math.min(220, (this.w * this.h) / 9000));
    const palette = ['236,230,216', '236,230,216', '236,230,216', '227,179,65', '111,198,207'];
    this.stars = Array.from({ length: count }, () => {
      const z = 0.25 + Math.random() * 0.75;
      return {
        x: Math.random() * this.w, y: Math.random() * this.h, z,
        r:  z * 1.3 + 0.2,
        a:  0.35 + z * 0.5,
        tw: 0.0006 + Math.random() * 0.0016,
        ph: Math.random() * Math.PI * 2,
        c:  palette[(Math.random() * palette.length) | 0],
      };
    });

    // ── orbit center from the .bg-contour element ────────────────────────────
    const r = document.querySelector('.bg-contour')?.getBoundingClientRect();
    this.ocx    = r ? r.left + r.width  / 2 : this.w * 0.82;
    this.ocy    = r ? r.top  + r.height / 2 : this.h * 0.18;
    this.oscale = r ? r.width / 2            : Math.min(this.w, this.h) * 0.5;

    // ── orbiters — build once; on subsequent resizes just update orbit radii ─
    if (this.orbiters.length === 0) {
      const orbColors: number[][] = [
        [236, 230, 216], [236, 230, 216], [236, 230, 216], [236, 230, 216],
        [236, 230, 216], [236, 230, 216],
        [227, 179, 65],  [227, 179, 65],  [227, 179, 65],
        [111, 198, 207], [111, 198, 207], [111, 198, 207],
        [236, 230, 216], [227, 179, 65],
      ];
      this.orbiters = Array.from({ length: 14 }, (_, i) => {
        const inc = 0.4 + Math.random() * 0.75;       // inclination 0.4..1.15 rad
        return {
          rr:      (0.18 + Math.random() * 0.74) * this.oscale,
          ang:     Math.random() * Math.PI * 2,
          spd:     (0.00004 + Math.random() * 0.00008) * (Math.random() < 0.5 ? 1 : -1),
          size:    2.5 + Math.random() * 2.5,          // 2.5..5 px
          c:       orbColors[i % orbColors.length],
          rot:     Math.random() * Math.PI * 2,
          rspd:    (0.0008 + Math.random() * 0.0012) * (Math.random() < 0.5 ? 1 : -1),
          inc,
          squash:  Math.cos(inc),
          tphase:  Math.random() * Math.PI * 2,
        };
      });
    } else {
      // Rescale radii proportionally when viewport changes
      const prevScale = this.oscale;
      for (const o of this.orbiters) {
        if (prevScale > 0) o.rr = o.rr / prevScale * this.oscale;
      }
    }

    // ── gems — build once ────────────────────────────────────────────────────
    if (this.gems.length === 0) {
      const gemBases: number[][] = [
        [227, 179, 65],
        [227, 179, 65],
        [236, 230, 216],
        [236, 230, 216],
        [227, 179, 65],
      ];
      this.gems = Array.from({ length: 5 }, (_, i) => ({
        x:     Math.random() * this.w,
        y:     Math.random() * this.h,
        vx:    -(0.008 + Math.random() * 0.012),
        vy:     0.010 + Math.random() * 0.016,
        R:      9 + Math.random() * 9,
        rot:    Math.random() * Math.PI * 2,
        rspd:  (0.0006 + Math.random() * 0.001) * (Math.random() < 0.5 ? 1 : -1),
        depth:  0.6 + Math.random() * 0.4,
        base:   gemBases[i],
      }));
    }
  }

  private spawnShoot(): void {
    const fromTop = Math.random() < 0.6;
    const x = fromTop ? Math.random() * this.w : this.w * (0.6 + Math.random() * 0.4);
    const y = fromTop ? -20 : Math.random() * this.h * 0.4;
    const sp = 0.5 + Math.random() * 0.4;
    this.shoots.push({
      x, y,
      vx: -(0.5 + Math.random() * 0.3) * sp,
      vy:  (0.55 + Math.random() * 0.35) * sp,
      life: 0,
      max: 900 + Math.random() * 700,
      len: 120 + Math.random() * 120,
      rot:  0,
      rspd: 0.004 + Math.random() * 0.004,   // 0.004..0.008 rad/ms
      size: 4 + Math.random() * 2,            // 4..6 px
    });
  }

  // ── drawStar: anti-shuriken sparkle/gem shape ─────────────────────────────
  // inner = 0.20 * R gives THIN spikes (not fat shuriken blades)
  // diffraction cross-rays + round halo → reads as twinkling gem/star
  private drawStar(
    ctx: CanvasRenderingContext2D,
    cx: number, cy: number,
    R: number, rot: number,
    base: number[],
    ornate: boolean,
  ): void {
    const inner = R * 0.20;
    const light = -Math.PI / 4;
    const pts: [number, number][] = [];
    for (let i = 0; i < 8; i++) {
      const a   = (i * Math.PI / 4) - Math.PI / 2;
      const rad = i % 2 === 0 ? R : inner;
      pts.push([Math.cos(a) * rad, Math.sin(a) * rad]);
    }

    ctx.save();
    ctx.translate(cx, cy);

    // round soft halo
    const halo = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 1.7);
    halo.addColorStop(0, `rgba(${base[0]},${base[1]},${base[2]},0.22)`);
    halo.addColorStop(1, `rgba(${base[0]},${base[1]},${base[2]},0)`);
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(0, 0, R * 1.7, 0, Math.PI * 2); ctx.fill();

    ctx.rotate(rot);

    // diffraction cross-rays — 4 thin bright lines out of the main spikes → "twinkle"
    ctx.save();
    for (let k = 0; k < 4; k++) {
      ctx.rotate(Math.PI / 2);
      const rg = ctx.createLinearGradient(0, 0, 0, -R * 1.6);
      rg.addColorStop(0, 'rgba(255,250,240,0.55)');
      rg.addColorStop(1, `rgba(${base[0]},${base[1]},${base[2]},0)`);
      ctx.strokeStyle = rg; ctx.lineWidth = 0.9; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -R * 1.6); ctx.stroke();
    }
    ctx.restore();

    // faceted body — deep relief (sharp light falloff, inner ratio 0.20)
    for (let i = 0; i < 8; i++) {
      const p1   = pts[i], p2 = pts[(i + 1) % 8];
      const midA = ((i + 0.5) * Math.PI / 4) - Math.PI / 2 + rot;
      const lit  = 0.10 + 0.90 * Math.pow(Math.max(0, Math.cos(midA - light)), 1.6);
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.closePath();
      ctx.fillStyle = `rgba(${base[0]},${base[1]},${base[2]},${lit.toFixed(3)})`;
      ctx.fill();
    }

    // inner layered star (rotated 45°, ~0.5 R) → stepped bevel = more relief
    ctx.save();
    ctx.rotate(Math.PI / 4);
    for (let i = 0; i < 8; i++) {
      const a1   = (i * Math.PI / 4) - Math.PI / 2;
      const a2   = ((i + 1) * Math.PI / 4) - Math.PI / 2;
      const r1   = (i % 2 === 0 ? R : inner) * 0.5;
      const r2   = ((i + 1) % 2 === 0 ? R : inner) * 0.5;
      const midA = ((i + 0.5) * Math.PI / 4) - Math.PI / 2 + rot + Math.PI / 4;
      const lit  = 0.20 + 0.80 * Math.max(0, Math.cos(midA - light));
      ctx.beginPath(); ctx.moveTo(0, 0);
      ctx.lineTo(Math.cos(a1) * r1, Math.sin(a1) * r1);
      ctx.lineTo(Math.cos(a2) * r2, Math.sin(a2) * r2);
      ctx.closePath();
      ctx.fillStyle = `rgba(255,250,240,${(lit * 0.5).toFixed(3)})`;
      ctx.fill();
    }
    ctx.restore();

    // specular core
    ctx.beginPath(); ctx.arc(0, 0, R * 0.16, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,250,240,0.95)'; ctx.fill();

    // edge ornaments (only for big/ornate gems): outline + beads at tips and notches
    if (ornate) {
      ctx.beginPath();
      for (let i = 0; i < 8; i++) { const p = pts[i]; i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]); }
      ctx.closePath();
      ctx.lineWidth = 0.8; ctx.strokeStyle = 'rgba(236,230,216,0.55)'; ctx.stroke();
      for (let i = 0; i < 8; i++) {
        const p  = pts[i];
        const br = (i % 2 === 0 ? R * 0.07 : R * 0.05);
        ctx.beginPath(); ctx.arc(p[0], p[1], br, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(236,230,216,0.7)'; ctx.fill();
      }
    }

    ctx.restore();
  }

  private draw(t: number, still: boolean): void {
    const dt  = this.last ? Math.min(t - this.last, 60) : 16;
    this.last = t;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);

    // Lerp parallax offsets and publish as CSS variables each frame
    if (!still) {
      this.px += (this.tpx - this.px) * 0.05;
      this.py += (this.tpy - this.py) * 0.05;
      document.documentElement.style.setProperty('--par-x', this.px.toFixed(4));
      document.documentElement.style.setProperty('--par-y', this.py.toFixed(4));
    }

    // Global precession for tilted orbits
    if (!still) {
      this.tilt += 0.00004 * dt;
    }

    // ── drifting star field ──────────────────────────────────────────────────
    for (const s of this.stars) {
      if (!still) {
        s.y += s.z * 0.12 * (dt / 16);
        s.x -= s.z * 0.05 * (dt / 16);
        if (s.y > this.h + 2) { s.y = -2; s.x = Math.random() * this.w; }
        if (s.x < -2) { s.x = this.w + 2; }
      }
      const tw    = still ? 1 : 0.7 + 0.3 * Math.sin(t * s.tw + s.ph);
      const drawX = s.x + this.px * s.z * 16;
      const drawY = s.y + this.py * s.z * 16;
      ctx.beginPath();
      ctx.fillStyle = `rgba(${s.c},${(s.a * tw).toFixed(3)})`;
      ctx.arc(drawX, drawY, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── shooting stars ───────────────────────────────────────────────────────
    if (!still) {
      this.nextShoot -= dt;
      if (this.nextShoot <= 0) { this.spawnShoot(); this.nextShoot = 3500 + Math.random() * 5000; }
      for (let i = this.shoots.length - 1; i >= 0; i--) {
        const sh = this.shoots[i];
        sh.life += dt;
        sh.x += sh.vx * dt; sh.y += sh.vy * dt;
        sh.rot += sh.rspd * dt;

        const p     = sh.life / sh.max;
        const alpha = Math.sin(Math.min(p, 1) * Math.PI) * 0.9;
        // trail tail ends BEHIND the sparkle head
        const tx    = sh.x - sh.vx * sh.len / Math.hypot(sh.vx, sh.vy);
        const ty    = sh.y - sh.vy * sh.len / Math.hypot(sh.vx, sh.vy);
        const g     = ctx.createLinearGradient(sh.x, sh.y, tx, ty);
        g.addColorStop(0,   `rgba(236,230,216,${alpha.toFixed(3)})`);
        g.addColorStop(0.4, `rgba(227,179,65,${(alpha * 0.5).toFixed(3)})`);
        g.addColorStop(1,   'rgba(227,179,65,0)');
        ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(sh.x, sh.y); ctx.lineTo(tx, ty); ctx.stroke();

        // spinning sparkle head at the leading point
        ctx.globalAlpha = alpha;
        this.drawStar(ctx, sh.x, sh.y, sh.size, sh.rot, [236, 230, 216], false);
        ctx.globalAlpha = 1;

        if (sh.life >= sh.max || sh.x < -50 || sh.y > this.h + 50) this.shoots.splice(i, 1);
      }
    }

    // ── orbiting sparkle stars — tilted, precessing 3D orbits ────────────────
    for (const o of this.orbiters) {
      if (!still) {
        o.ang += o.spd * dt;
        o.rot += o.rspd * dt;
      }
      // Project onto tilted ellipse with precession
      const ex    = Math.cos(o.ang) * o.rr;
      const ey    = Math.sin(o.ang) * o.rr * o.squash;      // inclination squash
      const ta    = this.tilt + o.tphase;                   // precessing orientation
      const rx    = ex * Math.cos(ta) - ey * Math.sin(ta);
      const ry    = ex * Math.sin(ta) + ey * Math.cos(ta);
      const depth = 0.55 + 0.45 * Math.sin(o.ang);          // front side → brighter/bigger
      const x     = this.ocx + rx + this.px * 8;
      const y     = this.ocy + ry + this.py * 8;

      ctx.globalAlpha = 0.35 + 0.65 * depth;
      this.drawStar(ctx, x, y, o.size * (0.7 + 0.6 * depth), o.rot, o.c, false);
      ctx.globalAlpha = 1;
    }

    // ── spinning faceted gem stars (big, ornate) ─────────────────────────────
    for (const g of this.gems) {
      if (!still) {
        g.x   += g.vx * dt;
        g.y   += g.vy * dt;
        // Wrap around edges
        if (g.x < -g.R * 3) g.x = this.w + g.R * 3;
        if (g.x > this.w + g.R * 3) g.x = -g.R * 3;
        if (g.y < -g.R * 3) g.y = this.h + g.R * 3;
        if (g.y > this.h + g.R * 3) g.y = -g.R * 3;
        g.rot += g.rspd * dt;
      }
      const drawX = g.x + this.px * g.depth * 22;
      const drawY = g.y + this.py * g.depth * 22;
      this.drawStar(ctx, drawX, drawY, g.R, g.rot, g.base, true);
    }
  }
}
