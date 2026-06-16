import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

interface UserOut { username: string; role: string; is_active: boolean; }

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.css',
})
export class AdminComponent {
  private http = inject(HttpClient);
  private base = environment.apiBase;

  users = signal<UserOut[]>([]);
  newUsername = '';
  newPassword = '';
  newRole = 'user';
  error = signal<string | null>(null);
  ok = signal<string | null>(null);

  ngOnInit(): void { this.refresh(); }

  refresh(): void {
    this.http.get<UserOut[]>(`${this.base}/admin/users`).subscribe({
      next: (u) => this.users.set(u),
      error: (e) => this.error.set(e?.error?.detail ?? 'Не удалось загрузить пользователей'),
    });
  }

  create(): void {
    this.error.set(null); this.ok.set(null);
    if (!this.newUsername || !this.newPassword) return;
    this.http.post(`${this.base}/admin/users`, {
      username: this.newUsername.trim(), password: this.newPassword, role: this.newRole,
    }).subscribe({
      next: () => {
        this.ok.set(`Создан пользователь ${this.newUsername}`);
        this.newUsername = ''; this.newPassword = ''; this.newRole = 'user';
        this.refresh();
      },
      error: (e) => this.error.set(e?.status === 409 ? 'Пользователь уже существует' : (e?.error?.detail ?? 'Ошибка создания')),
    });
  }

  setActive(u: UserOut, active: boolean): void {
    this.http.post(`${this.base}/admin/users/${encodeURIComponent(u.username)}/active?active=${active}`, {})
      .subscribe({ next: () => this.refresh(), error: () => this.refresh() });
  }
}
