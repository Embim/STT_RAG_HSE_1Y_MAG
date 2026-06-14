import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  AuthCheckResponse, ForwardRequest, ForwardResponse, SourceFilesResponse,
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
}
