import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  username = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  submit(): void {
    if (!this.username || !this.password || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.auth.login(this.username.trim(), this.password).subscribe({
      next: () => { this.loading.set(false); this.router.navigate(['/']); },
      error: (e) => {
        this.loading.set(false);
        this.error.set(e?.status === 401 ? 'Неверный логин или пароль' : (e?.error?.detail ?? 'Ошибка входа'));
      },
    });
  }
}
