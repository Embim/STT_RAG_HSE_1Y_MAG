import { Injectable, inject, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthUser, LoginResponse } from './auth.types';

const TOKEN_KEY = 'ds_token';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private base = environment.apiBase;
  private _user = signal<AuthUser | null>(null);

  readonly user = this._user.asReadonly();
  readonly isAuthenticated = computed(() => this._user() !== null);
  readonly isAdmin = computed(() => this._user()?.role === 'admin');

  token(): string | null { return localStorage.getItem(TOKEN_KEY); }
  hasToken(): boolean { return !!localStorage.getItem(TOKEN_KEY); }

  login(username: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.base}/auth/login`, { username, password }).pipe(
      tap((r) => { localStorage.setItem(TOKEN_KEY, r.access_token); this._user.set(r.user); }),
    );
  }

  loadMe(): Observable<AuthUser> {
    return this.http.get<AuthUser>(`${this.base}/auth/me`).pipe(tap((u) => this._user.set(u)));
  }

  logout(): void { localStorage.removeItem(TOKEN_KEY); this._user.set(null); }
}
