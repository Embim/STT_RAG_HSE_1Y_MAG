import { Routes } from '@angular/router';
import { authGuard, adminGuard } from './core/auth.guard';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./features/login/login.component').then(m => m.LoginComponent) },
  { path: '', canActivate: [authGuard], loadComponent: () => import('./features/search/search.component').then(m => m.SearchComponent) },
  { path: 'admin', canActivate: [adminGuard], loadComponent: () => import('./features/admin/admin.component').then(m => m.AdminComponent) },
  { path: 'ingest', canActivate: [authGuard], loadComponent: () => import('./features/ingest/ingest.component').then(m => m.IngestComponent) },
  { path: 'space', canActivate: [authGuard], loadComponent: () => import('./features/space/space.component').then(m => m.SpaceComponent) },
  { path: '**', redirectTo: '' },
];
