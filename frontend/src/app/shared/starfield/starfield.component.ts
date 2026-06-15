import {
  Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild, inject,
} from '@angular/core';

interface Star    { x: number; y: number; z: number; r: number; a: number; tw: number; ph: number; c: string; }
interface Shoot   { x: number; y: number; vx: number; vy: number; life: number; max: number; len: number; }
interface Orbiter { rr: number; ang: number; spd: number; size: number; c: string; }
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

    // ── orbiters — build once; on subsequent resizes just update orbit center ─
    if (this.orbiters.length === 0) {
      const orbColors = [
        '236,230,216', '236,230,216', '236,230,216', '236,230,216',
        '227,179,65',  '227,179,65',
        '111,198,207', '111,198,207',
      ];
      this.orbiters = Array.from({ length: 14 }, () => ({
        rr:   (0.18 + Math.random() * 0.74) /* 0.18..0.92 */ * this.oscale,
        ang:  Math.random() * Math.PI * 2,
        spd:  (0.00004 + Math.random() * 0.00008) * (Math.random() < 0.5 ? 1 : -1),
        size: 0.8 + Math.random() * 1.0,
        c:    orbColors[(Math.random() * orbColors.length) | 0],
      }));
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
    });
  }

  private drawGem(ctx: CanvasRenderingContext2D, cx: number, cy: number, R: number, rot: number, base: number[]): void {
    const inner      = R * 0.36;
    const lightAngle = -Math.PI / 4;            // light from upper-left
    const pts: [number, number][] = [];
    for (let i = 0; i < 8; i++) {
      const a   = (i * Math.PI / 4) - Math.PI / 2;   // start at top point
      const rad = i % 2 === 0 ? R : inner;
      pts.push([Math.cos(a) * rad, Math.sin(a) * rad]);
    }
    ctx.save();
    ctx.translate(cx, cy);
    // soft glow behind the gem
    const halo = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 1.8);
    halo.addColorStop(0, `rgba(${base[0]},${base[1]},${base[2]},0.18)`);
    halo.addColorStop(1, `rgba(${base[0]},${base[1]},${base[2]},0)`);
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(0, 0, R * 1.8, 0, Math.PI * 2); ctx.fill();
    ctx.rotate(rot);
    // 8 faceted triangles — brightness from facet normal vs light direction (relief)
    for (let i = 0; i < 8; i++) {
      const p1    = pts[i], p2 = pts[(i + 1) % 8];
      const midA  = ((i + 0.5) * Math.PI / 4) - Math.PI / 2 + rot;  // world-space facet angle
      const lit   = 0.32 + 0.68 * Math.max(0, Math.cos(midA - lightAngle));  // 0.32..1.0
      ctx.beginPath();
      ctx.moveTo(0, 0); ctx.lineTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.closePath();
      ctx.fillStyle = `rgba(${base[0]},${base[1]},${base[2]},${lit.toFixed(3)})`;
      ctx.fill();
    }
    // bright core
    ctx.beginPath(); ctx.arc(0, 0, R * 0.14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,250,240,0.95)';
    ctx.fill();
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
        const p     = sh.life / sh.max;
        const alpha = Math.sin(Math.min(p, 1) * Math.PI) * 0.9;
        const tx    = sh.x - sh.vx * sh.len / Math.hypot(sh.vx, sh.vy);
        const ty    = sh.y - sh.vy * sh.len / Math.hypot(sh.vx, sh.vy);
        const g     = ctx.createLinearGradient(sh.x, sh.y, tx, ty);
        g.addColorStop(0,   `rgba(236,230,216,${alpha.toFixed(3)})`);
        g.addColorStop(0.4, `rgba(227,179,65,${(alpha * 0.5).toFixed(3)})`);
        g.addColorStop(1,   'rgba(227,179,65,0)');
        ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(sh.x, sh.y); ctx.lineTo(tx, ty); ctx.stroke();
        if (sh.life >= sh.max || sh.x < -50 || sh.y > this.h + 50) this.shoots.splice(i, 1);
      }
    }

    // ── orbiting stars ───────────────────────────────────────────────────────
    for (const o of this.orbiters) {
      if (!still) o.ang += o.spd * dt;
      const ox = this.ocx + Math.cos(o.ang) * o.rr + this.px * 8;
      const oy = this.ocy + Math.sin(o.ang) * o.rr + this.py * 8;
      // Glint: small filled circle with a radial-gradient halo
      const glint = ctx.createRadialGradient(ox, oy, 0, ox, oy, o.size * 3.5);
      glint.addColorStop(0,   `rgba(${o.c},0.85)`);
      glint.addColorStop(0.4, `rgba(${o.c},0.35)`);
      glint.addColorStop(1,   `rgba(${o.c},0)`);
      ctx.beginPath();
      ctx.fillStyle = glint;
      ctx.arc(ox, oy, o.size * 3.5, 0, Math.PI * 2);
      ctx.fill();
      // Solid core dot
      ctx.beginPath();
      ctx.fillStyle = `rgba(${o.c},0.9)`;
      ctx.arc(ox, oy, o.size * 0.7, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── spinning faceted gem stars ───────────────────────────────────────────
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
      this.drawGem(ctx, drawX, drawY, g.R, g.rot, g.base);
    }
  }
}
