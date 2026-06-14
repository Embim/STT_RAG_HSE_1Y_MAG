import { Component, inject, signal, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { ApiService } from '../../core/api.service';
import { ForwardResponse, RetrievedDoc } from '../../core/api.types';
import { formatTimecode, deepLink } from '../../core/timecode.util';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './search.component.html',
  styleUrl: './search.component.css',
})
export class SearchComponent implements OnInit {
  private api = inject(ApiService);

  question = '';
  topK = 5;
  threshold = 0.5;
  useRewrite = true;
  sourceTitle = '';

  sources = signal<string[]>([]);
  loading = signal(false);
  error = signal<string | null>(null);
  result = signal<ForwardResponse | null>(null);

  ngOnInit(): void {
    this.api.sourceFiles().subscribe({
      next: (r) => this.sources.set(r.files ?? []),
      error: () => {},
    });
  }

  search(): void {
    const q = this.question.trim();
    if (!q || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);
    this.api.forward({
      question: q,
      top_k: this.topK,
      similarity_threshold: this.threshold,
      use_rewrite: this.useRewrite,
      source_title: this.sourceTitle || null,
    }).subscribe({
      next: (r) => {
        if (!r || !r.answer) {
          this.error.set('Модель не вернула ответ — попробуйте переформулировать вопрос или снизить порог.');
        } else {
          this.result.set(r);
        }
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(e?.error?.detail ?? e?.message ?? 'Ошибка запроса');
        this.loading.set(false);
      },
    });
  }

  answerHtml(md: string): string {
    return DOMPurify.sanitize(marked.parse(md) as string);
  }

  timecode = formatTimecode;
  link = deepLink;

  docTitle(d: RetrievedDoc, i: number): string {
    return d.metadata?.title || d.metadata?.source_file_name || `Фрагмент ${i + 1}`;
  }
}
