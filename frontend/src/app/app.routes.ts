import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./features/search/search.component').then(m => m.SearchComponent),
  },
];
