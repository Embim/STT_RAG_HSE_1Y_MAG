export interface AuthUser { username: string; role: string; }
export interface LoginResponse { access_token: string; token_type: string; user: AuthUser; }
