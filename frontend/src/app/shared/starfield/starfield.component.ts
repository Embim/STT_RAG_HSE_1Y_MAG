import {
  Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild, inject,
} from '@angular/core';

interface Star    { x: number; y: number; z: number; r: number; a: number; tw: number; ph: number; c: string; }
interface Shoot   { x: number; y: number; vx: number; vy: number; life: number; max: number; len: number; rot: number; rspd: number; size: number; ax: number; ay: number; axs: number; ays: number; rx: number; ry: number; rz: number; rxs: number; rys: number; rzs: number; }
interface Orbiter { rr: number; ang: number; spd: number; size: number; c: number[]; inc: number; squash: number; tphase: number; rx: number; ry: number; rz: number; rxs: number; rys: number; rzs: number; }
interface Gem     { x: number; y: number; vx: number; vy: number; R: number; depth: number; base: number[]; rx: number; ry: number; rz: number; rxs: number; rys: number; rzs: number; }

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
      const rSign = () => Math.random() < 0.5 ? 1 : -1;
      this.orbiters = Array.from({ length: 14 }, (_, i) => {
        const inc = 0.4 + Math.random() * 0.75;       // inclination 0.4..1.15 rad
        return {
          rr:      (0.18 + Math.random() * 0.74) * this.oscale,
          ang:     Math.random() * Math.PI * 2,
          spd:     (0.00004 + Math.random() * 0.00008) * rSign(),
          size:    5 + Math.random() * 3,              // 5..8 px — larger so 3D reads well
          c:       orbColors[i % orbColors.length],
          inc,
          squash:  Math.cos(inc),
          tphase:  Math.random() * Math.PI * 2,
          rx:      Math.random() * Math.PI * 2,
          ry:      Math.random() * Math.PI * 2,
          rz:      Math.random() * Math.PI * 2,
          rxs:     (0.0008 + Math.random() * 0.001) * rSign(),
          rys:     (0.0008 + Math.random() * 0.001) * rSign(),
          rzs:     (0.0008 + Math.random() * 0.001) * rSign(),
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
      const gSign = () => Math.random() < 0.5 ? 1 : -1;
      this.gems = Array.from({ length: 5 }, (_, i) => ({
        x:     Math.random() * this.w,
        y:     Math.random() * this.h,
        vx:    -(0.008 + Math.random() * 0.012),
        vy:     0.010 + Math.random() * 0.016,
        R:      9 + Math.random() * 9,
        depth:  0.6 + Math.random() * 0.4,
        base:   gemBases[i],
        rx:     Math.random() * Math.PI * 2,
        ry:     Math.random() * Math.PI * 2,
        rz:     Math.random() * Math.PI * 2,
        rxs:    (0.0006 + Math.random() * 0.0008) * gSign(),
        rys:    (0.0006 + Math.random() * 0.0008) * gSign(),
        rzs:    (0.0006 + Math.random() * 0.0008) * gSign(),
      }));
    }
  }

  private spawnShoot(): void {
    const fromTop = Math.random() < 0.6;
    const x = fromTop ? Math.random() * this.w : this.w * (0.6 + Math.random() * 0.4);
    const y = fromTop ? -20 : Math.random() * this.h * 0.4;
    const sp = 0.5 + Math.random() * 0.4;
    const rSign = () => Math.random() < 0.5 ? 1 : -1;
    this.shoots.push({
      x, y,
      vx: -(0.5 + Math.random() * 0.3) * sp,
      vy:  (0.55 + Math.random() * 0.35) * sp,
      life: 0,
      max: 900 + Math.random() * 700,
      len: 120 + Math.random() * 120,
      rot:  0,
      rspd: 0.004 + Math.random() * 0.004,
      size: 4 + Math.random() * 2,            // 4..6 px → 3D head radius ≈ 7.6..11.4
      ax:   Math.random() * Math.PI * 2,
      ay:   Math.random() * Math.PI * 2,
      axs:  (0.0009 + Math.random() * 0.0007) * rSign(),
      ays:  (0.0009 + Math.random() * 0.0007) * rSign(),
      // 3D rotation angles and per-axis spin speeds (rad/ms)
      rx:  Math.random() * Math.PI * 2,
      ry:  Math.random() * Math.PI * 2,
      rz:  Math.random() * Math.PI * 2,
      rxs: (0.0010 + Math.random() * 0.0016) * rSign(),
      rys: (0.0010 + Math.random() * 0.0016) * rSign(),
      rzs: (0.0010 + Math.random() * 0.0016) * rSign(),
    });
  }

  // ── drawStar3D: software-rendered 3D stellated (dimpled) octahedron ─────────
  private drawStar3D(
    ctx: CanvasRenderingContext2D,
    cx: number, cy: number,
    R: number,
    rx: number, ry: number, rz: number,
    base: number[],
  ): void {
    // ── stellated octahedron geometry ────────────────────────────────────────
    const TIP   = 1.0;
    const INSET = 0.40;  // distance of per-face inset vertex from origin.
                         // < 0.577 (natural oct face dist) → concave dimple. TUNABLE.

    const tips: number[][] = [
      [ TIP,   0,   0], [-TIP,   0,   0],
      [   0, TIP,   0], [   0, -TIP,   0],
      [   0,   0, TIP], [   0,   0, -TIP],
    ];
    // 8 octahedron faces (one per octant), each as index triple into tips[]
    const oct: number[][] = [
      [0,2,4],[2,1,4],[1,3,4],[3,0,4],
      [2,0,5],[1,2,5],[3,1,5],[0,3,5],
    ];
    const V: number[][] = tips.map(t => t.slice());
    const F: number[][] = [];
    for (const fc of oct) {
      const a = tips[fc[0]], b = tips[fc[1]], c = tips[fc[2]];
      const sx = a[0]+b[0]+c[0], sy = a[1]+b[1]+c[1], sz = a[2]+b[2]+c[2];
      const L  = Math.hypot(sx, sy, sz) || 1;
      const m  = [sx/L*INSET, sy/L*INSET, sz/L*INSET]; // face-center vertex pulled toward origin
      const mi = V.length; V.push(m);
      // split flat face into 3 triangles meeting at the inset center → concave dimple
      F.push([fc[0], fc[1], mi], [fc[1], fc[2], mi], [fc[2], fc[0], mi]);
    }

    // ── rotation matrix & perspective projection ─────────────────────────────
    const cax=Math.cos(rx),sax=Math.sin(rx),cay=Math.cos(ry),say=Math.sin(ry),caz=Math.cos(rz),saz=Math.sin(rz);
    const rot = (p: number[]): number[] => {
      let x=p[0],y=p[1],z=p[2];
      let y1=y*cax - z*sax, z1=y*sax + z*cax; y=y1; z=z1;            // X rotation
      let x1=x*cay + z*say, z2=-x*say + z*cay; x=x1; z=z2;            // Y rotation
      let x2=x*caz - y*saz, y2=x*saz + y*caz; x=x2; y=y2;            // Z rotation
      return [x,y,z];
    };
    const RV = V.map(rot);
    const fdist = 3.4;                                                 // perspective camera distance
    const proj = (p: number[]): number[] => { const s = fdist / (fdist - p[2]); return [p[0]*s, p[1]*s]; };
    const light = [-0.40, -0.55, 0.73];                               // upper-left, toward viewer (≈unit)
    const dot = (a: number[], b: number[]) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

    ctx.save();
    ctx.translate(cx, cy);

    // soft round halo
    const halo = ctx.createRadialGradient(0,0,0,0,0,R*2.0);
    halo.addColorStop(0, `rgba(${base[0]},${base[1]},${base[2]},0.22)`);
    halo.addColorStop(1, `rgba(${base[0]},${base[1]},${base[2]},0)`);
    ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(0,0,R*2.0,0,Math.PI*2); ctx.fill();

    // radiant glow spikes from the 6 octahedron axis tips (V[0..5])
    for (let i = 0; i < 6; i++) {
      const tip = RV[i];
      const tproj = proj([tip[0] * 1.7, tip[1] * 1.7, tip[2]]);
      const base0 = proj(tip);
      const facing = Math.max(0, (tip[2] + 1) / 2);                  // front-facing tips brighter
      const g = ctx.createLinearGradient(base0[0]*R, base0[1]*R, tproj[0]*R, tproj[1]*R);
      g.addColorStop(0, `rgba(255,250,240,${(0.45*facing+0.08).toFixed(3)})`);
      g.addColorStop(1, `rgba(${base[0]},${base[1]},${base[2]},0)`);
      ctx.strokeStyle = g; ctx.lineWidth = 0.8; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(base0[0]*R, base0[1]*R); ctx.lineTo(tproj[0]*R, tproj[1]*R); ctx.stroke();
    }

    // faces: outward normal, back-face cull, diffuse shade, draw front faces
    for (const f of F) {
      const a=RV[f[0]], b=RV[f[1]], c=RV[f[2]];
      const u=[b[0]-a[0],b[1]-a[1],b[2]-a[2]], v=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
      let n=[u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
      const nl=Math.hypot(n[0],n[1],n[2])||1; n=[n[0]/nl,n[1]/nl,n[2]/nl];
      const cen=[(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3,(a[2]+b[2]+c[2])/3];
      if (dot(n,cen) < 0) n=[-n[0],-n[1],-n[2]];                     // orient outward
      if (n[2] <= 0) continue;                                       // back-face cull (camera looks down +z)
      const lit = 0.16 + 0.84 * Math.max(0, dot(n, light));
      const pa=proj(a), pb=proj(b), pc=proj(c);
      ctx.beginPath(); ctx.moveTo(pa[0]*R,pa[1]*R); ctx.lineTo(pb[0]*R,pb[1]*R); ctx.lineTo(pc[0]*R,pc[1]*R); ctx.closePath();
      ctx.fillStyle = `rgba(${base[0]},${base[1]},${base[2]},${lit.toFixed(3)})`;
      ctx.fill();
      ctx.lineWidth=0.4; ctx.strokeStyle=`rgba(255,250,240,${(lit*0.3).toFixed(3)})`; ctx.stroke();
    }

    // bright specular core
    ctx.beginPath(); ctx.arc(0,0,R*0.16,0,Math.PI*2);
    ctx.fillStyle='rgba(255,250,240,0.95)'; ctx.fill();
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
        sh.ax += sh.axs * dt; sh.ay += sh.ays * dt;
        // advance 3D rotation angles
        sh.rx += sh.rxs * dt; sh.ry += sh.rys * dt; sh.rz += sh.rzs * dt;

        const p     = sh.life / sh.max;
        const alpha = Math.sin(Math.min(p, 1) * Math.PI) * 0.9;
        // trail tail ends BEHIND the 3D head
        const tx    = sh.x - sh.vx * sh.len / Math.hypot(sh.vx, sh.vy);
        const ty    = sh.y - sh.vy * sh.len / Math.hypot(sh.vx, sh.vy);
        const g     = ctx.createLinearGradient(sh.x, sh.y, tx, ty);
        g.addColorStop(0,   `rgba(236,230,216,${alpha.toFixed(3)})`);
        g.addColorStop(0.4, `rgba(227,179,65,${(alpha * 0.5).toFixed(3)})`);
        g.addColorStop(1,   'rgba(227,179,65,0)');
        ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(sh.x, sh.y); ctx.lineTo(tx, ty); ctx.stroke();

        // 3D rotating star bipyramid head at the leading point
        ctx.globalAlpha = alpha;
        this.drawStar3D(ctx, sh.x, sh.y, (sh.size ?? 5) * 1.9, sh.rx, sh.ry, sh.rz, [236, 230, 216]);
        ctx.globalAlpha = 1;

        if (sh.life >= sh.max || sh.x < -50 || sh.y > this.h + 50) this.shoots.splice(i, 1);
      }
    }

    // ── orbiting sparkle stars — tilted, precessing 3D orbits ────────────────
    for (const o of this.orbiters) {
      if (!still) {
        o.ang += o.spd * dt;
        o.rx  += o.rxs * dt;
        o.ry  += o.rys * dt;
        o.rz  += o.rzs * dt;
      }
      // Project onto tilted ellipse with precession
      const ex    = Math.cos(o.ang) * o.rr;
      const ey    = Math.sin(o.ang) * o.rr * o.squash;      // inclination squash
      const ta    = this.tilt + o.tphase;                   // precessing orientation
      const orx   = ex * Math.cos(ta) - ey * Math.sin(ta);
      const ory   = ex * Math.sin(ta) + ey * Math.cos(ta);
      const depth = 0.55 + 0.45 * Math.sin(o.ang);          // front side → brighter/bigger
      const x     = this.ocx + orx + this.px * 8;
      const y     = this.ocy + ory + this.py * 8;
      const headR = o.size * (0.8 + 0.7 * depth);

      ctx.globalAlpha = 0.35 + 0.65 * depth;
      this.drawStar3D(ctx, x, y, headR, o.rx, o.ry, o.rz, o.c);
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
        g.rx  += g.rxs * dt;
        g.ry  += g.rys * dt;
        g.rz  += g.rzs * dt;
      }
      const drawX = g.x + this.px * g.depth * 22;
      const drawY = g.y + this.py * g.depth * 22;
      this.drawStar3D(ctx, drawX, drawY, g.R, g.rx, g.ry, g.rz, g.base);
    }
  }
}
