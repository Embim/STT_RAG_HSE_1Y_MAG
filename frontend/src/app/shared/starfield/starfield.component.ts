import {
  Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild, inject,
} from '@angular/core';

interface Star { x: number; y: number; z: number; r: number; a: number; tw: number; ph: number; c: string; }
interface Shoot { x: number; y: number; vx: number; vy: number; life: number; max: number; len: number; }

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
  private zone = inject(NgZone);
  private ctx!: CanvasRenderingContext2D;
  private raf = 0;
  private w = 0; private h = 0; private dpr = 1;
  private stars: Star[] = [];
  private shoots: Shoot[] = [];
  private nextShoot = 2000;
  private last = 0;
  private reduced = false;
  private onResize = () => this.resize();

  ngAfterViewInit(): void {
    this.ctx = this.cv.nativeElement.getContext('2d')!;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.resize();
    window.addEventListener('resize', this.onResize);
    if (this.reduced) { this.draw(0, true); return; }
    this.zone.runOutsideAngular(() => {
      const loop = (t: number) => { this.draw(t, false); this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
  }

  private resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    const c = this.cv.nativeElement;
    c.width = Math.floor(this.w * this.dpr);
    c.height = Math.floor(this.h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const count = Math.round(Math.min(220, (this.w * this.h) / 9000));
    const palette = ['236,230,216', '236,230,216', '236,230,216', '227,179,65', '111,198,207'];
    this.stars = Array.from({ length: count }, () => {
      const z = 0.25 + Math.random() * 0.75;
      return {
        x: Math.random() * this.w, y: Math.random() * this.h, z,
        r: z * 1.3 + 0.2, a: 0.35 + z * 0.5,
        tw: 0.0006 + Math.random() * 0.0016, ph: Math.random() * Math.PI * 2,
        c: palette[(Math.random() * palette.length) | 0],
      };
    });
  }

  private spawnShoot(): void {
    const fromTop = Math.random() < 0.6;
    const x = fromTop ? Math.random() * this.w : this.w * (0.6 + Math.random() * 0.4);
    const y = fromTop ? -20 : Math.random() * this.h * 0.4;
    const sp = 0.5 + Math.random() * 0.4;
    this.shoots.push({ x, y, vx: -(0.5 + Math.random() * 0.3) * sp, vy: (0.55 + Math.random() * 0.35) * sp, life: 0, max: 900 + Math.random() * 700, len: 120 + Math.random() * 120 });
  }

  private draw(t: number, still: boolean): void {
    const dt = this.last ? Math.min(t - this.last, 60) : 16;
    this.last = t;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);

    for (const s of this.stars) {
      if (!still) {
        s.y += s.z * 0.12 * (dt / 16);
        s.x -= s.z * 0.05 * (dt / 16);
        if (s.y > this.h + 2) { s.y = -2; s.x = Math.random() * this.w; }
        if (s.x < -2) { s.x = this.w + 2; }
      }
      const tw = still ? 1 : 0.7 + 0.3 * Math.sin(t * s.tw + s.ph);
      ctx.beginPath();
      ctx.fillStyle = `rgba(${s.c},${(s.a * tw).toFixed(3)})`;
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    if (!still) {
      this.nextShoot -= dt;
      if (this.nextShoot <= 0) { this.spawnShoot(); this.nextShoot = 3500 + Math.random() * 5000; }
      for (let i = this.shoots.length - 1; i >= 0; i--) {
        const sh = this.shoots[i];
        sh.life += dt;
        sh.x += sh.vx * dt; sh.y += sh.vy * dt;
        const p = sh.life / sh.max;
        const alpha = Math.sin(Math.min(p, 1) * Math.PI) * 0.9;
        const tx = sh.x - sh.vx * sh.len / Math.hypot(sh.vx, sh.vy);
        const ty = sh.y - sh.vy * sh.len / Math.hypot(sh.vx, sh.vy);
        const g = ctx.createLinearGradient(sh.x, sh.y, tx, ty);
        g.addColorStop(0, `rgba(236,230,216,${alpha.toFixed(3)})`);
        g.addColorStop(0.4, `rgba(227,179,65,${(alpha * 0.5).toFixed(3)})`);
        g.addColorStop(1, 'rgba(227,179,65,0)');
        ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(sh.x, sh.y); ctx.lineTo(tx, ty); ctx.stroke();
        if (sh.life >= sh.max || sh.x < -50 || sh.y > this.h + 50) this.shoots.splice(i, 1);
      }
    }
  }
}
