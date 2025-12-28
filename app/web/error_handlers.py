"""
Custom error handlers to prevent information disclosure.
Provides generic error pages without exposing system details.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
import os

logger = logging.getLogger(__name__)

# Templates for error pages
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


async def generic_error_handler(request: Request, exc: Exception) -> HTMLResponse:
    """
    Generic error handler that prevents information disclosure.
    
    - Logs detailed error server-side
    - Returns user-friendly error page without system details
    - Does not expose stack traces or file paths
    """
    # Log detailed error server-side for debugging
    logger.error(
        f"Unhandled error: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    # Return generic error page to user
    # Accept both HTML and JSON requests
    accept_header = request.headers.get("accept", "")
    
    if "application/json" in accept_header:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )
    
    # Return HTML error page
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 500,
            "title": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        },
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


async def not_found_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Handle 404 Not Found errors with custom page."""
    logger.warning(f"404 Not Found: {request.url.path}")
    
    accept_header = request.headers.get("accept", "")
    
    if "application/json" in accept_header:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found."
            }
        )
    
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 404,
            "title": "Page Not Found",
            "message": "The page you're looking for doesn't exist."
        },
        status_code=status.HTTP_404_NOT_FOUND
    )


async def forbidden_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Handle 403 Forbidden errors."""
    logger.warning(
        f"403 Forbidden: {request.url.path} from {request.client.host if request.client else 'unknown'}"
    )
    
    accept_header = request.headers.get("accept", "")
    
    if "application/json" in accept_header:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Forbidden",
                "message": "You don't have permission to access this resource."
            }
        )
    
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 403,
            "title": "Access Denied",
            "message": "You don't have permission to access this resource."
        },
        status_code=status.HTTP_403_FORBIDDEN
    )


async def unauthorized_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Handle 401 Unauthorized errors."""
    logger.warning(
        f"401 Unauthorized: {request.url.path} from {request.client.host if request.client else 'unknown'}"
    )
    
    accept_header = request.headers.get("accept", "")
    
    if "application/json" in accept_header:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Unauthorized",
                "message": "Authentication is required to access this resource."
            }
        )
    
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 401,
            "title": "Authentication Required",
            "message": "Please log in to access this resource."
        },
        status_code=status.HTTP_401_UNAUTHORIZED
    )


def configure_error_handlers(app):
    """
    Configure custom error handlers for the FastAPI application.
    
    This prevents information disclosure by:
    - Hiding stack traces from users
    - Providing generic error messages
    - Logging detailed errors server-side only
    - Not exposing framework/library versions
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    # Handle validation errors (400)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad Request",
                "message": "The request data is invalid."
            }
        )
    
    # Handle HTTP exceptions
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return await not_found_handler(request, exc)
        elif exc.status_code == 403:
            return await forbidden_handler(request, exc)
        elif exc.status_code == 401:
            return await unauthorized_handler(request, exc)
        
        # Generic handler for other HTTP errors
        accept_header = request.headers.get("accept", "")
        
        if "application/json" in accept_header:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail if isinstance(exc.detail, str) else "Error",
                    "message": exc.detail if isinstance(exc.detail, str) else "An error occurred."
                }
            )
        
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": exc.status_code,
                "title": f"Error {exc.status_code}",
                "message": exc.detail if isinstance(exc.detail, str) else "An error occurred."
            },
            status_code=exc.status_code
        )
    
    # Handle all other exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return await generic_error_handler(request, exc)
