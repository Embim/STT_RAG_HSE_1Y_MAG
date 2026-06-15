import { Component, inject } from '@angular/core';
import { RouterOutlet, RouterLink, Router } from '@angular/router';
import { AuthService } from './core/auth.service';
import { StarfieldComponent } from './shared/starfield/starfield.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, StarfieldComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  auth = inject(AuthService);
  private router = inject(Router);
  logout(): void { this.auth.logout(); this.router.navigate(['/login']); }
}
