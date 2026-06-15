from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from system.auth import service
from system.auth.deps import get_current_user, require_admin
from system.auth.models import User
from system.auth.security import create_access_token

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserOut(BaseModel):
    username: str
    role: str
    is_active: bool


@router.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest):
    user = service.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_access_token(user.username, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"username": user.username, "role": user.role},
    }


@router.get("/auth/me", tags=["Auth"])
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


@router.post("/admin/users", response_model=UserOut, tags=["Admin"])
def admin_create_user(req: CreateUserRequest, _: User = Depends(require_admin)):
    if req.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="invalid role")
    try:
        u = service.create_user(req.username, req.password, req.role)
    except ValueError:
        raise HTTPException(status_code=409, detail="user already exists")
    return UserOut(username=u.username, role=u.role, is_active=u.is_active)


@router.get("/admin/users", response_model=list[UserOut], tags=["Admin"])
def admin_list_users(_: User = Depends(require_admin)):
    return [UserOut(username=u.username, role=u.role, is_active=u.is_active) for u in service.list_users()]


@router.post("/admin/users/{username}/active", response_model=UserOut, tags=["Admin"])
def admin_set_active(username: str, active: bool, _: User = Depends(require_admin)):
    u = service.set_active(username, active)
    if not u:
        raise HTTPException(status_code=404, detail="not found")
    return UserOut(username=u.username, role=u.role, is_active=u.is_active)
