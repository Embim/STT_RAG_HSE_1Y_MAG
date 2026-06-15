import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpEvent, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  AuthCheckResponse, ForwardRequest, ForwardResponse, SourceFilesResponse,
  IngestJob, JobAccepted, AsrModelsResponse, EmbeddingMapResponse, LocateResponse,
} from './api.types';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  private base = environment.apiBase; // '' in prod (same origin behind nginx)

  authCheck(): Observable<AuthCheckResponse> {
    return this.http.get<AuthCheckResponse>(`${this.base}/auth-check`);
  }

  sourceFiles(): Observable<SourceFilesResponse> {
    return this.http.get<SourceFilesResponse>(`${this.base}/source-files`);
  }

  forward(req: ForwardRequest): Observable<ForwardResponse> {
    return this.http.post<ForwardResponse>(`${this.base}/forward`, req);
  }

  ingestYoutube(body: { url: string; export_txt?: boolean; export_json?: boolean; use_ocr?: boolean; keep_audio?: boolean; asr_model?: string }): Observable<JobAccepted> {
    return this.http.post<JobAccepted>(`${this.base}/ingest`, body);
  }

  ingestUpload(files: File[], opts: { export_txt?: boolean; export_json?: boolean; keep_audio?: boolean; use_ocr?: boolean; asr_model?: string }): Observable<HttpEvent<JobAccepted>> {
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    let params = new HttpParams();
    if (opts.export_txt) params = params.set('export_txt', 'true');
    if (opts.export_json) params = params.set('export_json', 'true');
    if (opts.keep_audio) params = params.set('keep_audio', 'true');
    if (opts.use_ocr) params = params.set('use_ocr', 'true');
    if (opts.asr_model) params = params.set('asr_model', opts.asr_model);
    return this.http.post<JobAccepted>(`${this.base}/ingest-upload`, form, { params, reportProgress: true, observe: 'events' });
  }

  ingestStatus(jobId: string): Observable<IngestJob> {
    return this.http.get<IngestJob>(`${this.base}/ingest-status/${jobId}`);
  }

  asrModels(): Observable<AsrModelsResponse> {
    return this.http.get<AsrModelsResponse>(`${this.base}/asr-models`);
  }

  embeddingMap(force = false): Observable<EmbeddingMapResponse> {
    const url = `${this.base}/embedding-map${force ? '?force=true' : ''}`;
    return this.http.get<EmbeddingMapResponse>(url);
  }

  locateInMap(question: string, top_k = 8): Observable<LocateResponse> {
    return this.http.post<LocateResponse>(`${this.base}/embedding-map/locate`, { question, top_k });
  }
}
