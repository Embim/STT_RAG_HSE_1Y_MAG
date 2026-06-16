import { Component, inject, signal, OnInit, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpEventType } from '@angular/common/http';
import { Subscription, interval, switchMap } from 'rxjs';
import { ApiService } from '../../core/api.service';
import { IngestJob, AsrModelInfo } from '../../core/api.types';

@Component({
  selector: 'app-ingest',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './ingest.component.html',
  styleUrl: './ingest.component.css',
})
export class IngestComponent implements OnInit, OnDestroy {
  private api = inject(ApiService);

  // asr model catalog
  asrModels = signal<AsrModelInfo[]>([]);
  asrActive = signal<string | null>(null);
  asrAutoswap = signal(false);
  ytAsrModel = '';
  upAsrModel = '';

  // youtube
  ytUrl = '';
  ytTxt = false; ytJson = false; ytKeepAudio = false; ytUseOcr = false;
  // upload
  files: File[] = [];
  upTxt = false; upJson = false; upKeepAudio = false; upUseOcr = false;
  uploadPct = signal(0);

  busy = signal(false);
  error = signal<string | null>(null);
  job = signal<IngestJob | null>(null);
  private poll?: Subscription;

  ngOnInit(): void {
    this.api.asrModels().subscribe({
      next: (r) => {
        this.asrModels.set(r.models);
        this.asrActive.set(r.active);
        this.asrAutoswap.set(r.autoswap);
        // По умолчанию — то, что уже поднято (active), иначе серверный default.
        // Так пользователь не запускает тяжёлый свап случайно.
        const initial = r.active ?? r.default;
        this.ytAsrModel = initial;
        this.upAsrModel = initial;
      },
      error: () => { /* каталог не критичен — просто без выбора модели */ },
    });
  }

  /** Инфо о модели по ключу (для описания «когда какую»). */
  modelInfo(key: string): AsrModelInfo | undefined {
    return this.asrModels().find((m) => m.key === key);
  }

  /** Свап займёт время: автосвап включён и выбрана не та, что сейчас активна. */
  willSwap(key: string): boolean {
    return this.asrAutoswap() && !!this.asrActive() && key !== this.asrActive();
  }

  /** Можно ли выбрать модель: без авто-свапа обслуживается только активная. */
  selectable(m: AsrModelInfo): boolean {
    if (!m.available) return false;
    if (this.asrAutoswap()) return true;
    return !this.asrActive() || m.key === this.asrActive();
  }

  /** Без авто-свапа доступна только одна модель — показываем пояснение. */
  get autoswapOff(): boolean {
    return !this.asrAutoswap() && this.asrModels().length > 1;
  }

  submitYoutube(): void {
    const url = this.ytUrl.trim();
    if (!url || this.busy()) return;
    this.reset();
    this.busy.set(true);
    this.api.ingestYoutube({ url, export_txt: this.ytTxt, export_json: this.ytJson, use_ocr: this.ytUseOcr, keep_audio: this.ytKeepAudio, asr_model: this.ytAsrModel || undefined }).subscribe({
      next: (r) => { this.busy.set(false); this.startPolling(r.job_id); },
      error: (e) => { this.busy.set(false); this.error.set(e?.status === 400 ? 'Недопустимый URL (разрешены YouTube/Vimeo)' : (e?.error?.detail ?? 'Ошибка запроса')); },
    });
  }

  onFiles(e: Event): void {
    const input = e.target as HTMLInputElement;
    this.files = input.files ? Array.from(input.files) : [];
  }

  submitUpload(): void {
    if (!this.files.length || this.busy()) return;
    this.reset();
    this.busy.set(true);
    this.uploadPct.set(0);
    this.api.ingestUpload(this.files, { export_txt: this.upTxt, export_json: this.upJson, keep_audio: this.upKeepAudio, use_ocr: this.upUseOcr, asr_model: this.upAsrModel || undefined }).subscribe({
      next: (ev) => {
        if (ev.type === HttpEventType.UploadProgress && ev.total) {
          this.uploadPct.set(Math.round(100 * ev.loaded / ev.total));
        } else if (ev.type === HttpEventType.Response && ev.body) {
          this.busy.set(false);
          this.startPolling(ev.body.job_id);
        }
      },
      error: (e) => { this.busy.set(false); this.error.set(e?.status === 413 ? 'Файл слишком большой (макс 500 МБ)' : (e?.error?.detail ?? 'Ошибка загрузки')); },
    });
  }

  private startPolling(jobId: string): void {
    this.poll?.unsubscribe();
    this.poll = interval(2000).pipe(switchMap(() => this.api.ingestStatus(jobId))).subscribe({
      next: (j) => { this.job.set(j); if (j.status === 'done' || j.status === 'error') this.poll?.unsubscribe(); },
      error: () => this.poll?.unsubscribe(),
    });
  }

  private reset(): void {
    this.error.set(null);
    this.job.set(null);
    this.poll?.unsubscribe();
  }

  ngOnDestroy(): void { this.poll?.unsubscribe(); }
}
