"""Simple Dependency Injection Container."""
from typing import Any, Callable, Dict, Type, TypeVar, Optional, get_type_hints
from threading import Lock
from enum import Enum
import functools
import inspect

T = TypeVar('T')


class Lifetime(str, Enum):
    """Service lifetime."""
    TRANSIENT = "transient"
    SINGLETON = "singleton"


class DIContainer:
    """Dependency Injection Container with singleton and transient lifetime support."""

    def __init__(self) -> None:
        self._factories: Dict[Type[Any], Callable[[], Any]] = {}
        self._singletons: Dict[Type[Any], Any] = {}
        self._lifetimes: Dict[Type[Any], Lifetime] = {}
        self._lock = Lock()

    def register_transient(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a service with transient lifetime (new instance per resolve)."""
        with self._lock:
            self._factories[interface] = factory
            self._lifetimes[interface] = Lifetime.TRANSIENT
            self._singletons.pop(interface, None)

    def register_singleton(self, interface: Type[T], instance_or_factory: Any) -> None:
        """Register a service with singleton lifetime.

        Args:
            interface: The type to register.
            instance_or_factory: Either an existing instance or a factory callable.
                If callable, it will be invoked lazily on first resolve.
        """
        with self._lock:
            if callable(instance_or_factory) and not isinstance(instance_or_factory, type):
                self._factories[interface] = instance_or_factory
            else:
                self._singletons[interface] = instance_or_factory
                self._factories[interface] = lambda: instance_or_factory
            self._lifetimes[interface] = Lifetime.SINGLETON

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service by its interface type.

        Raises:
            ValueError: If the interface is not registered.
        """
        with self._lock:
            if interface in self._singletons:
                return self._singletons[interface]

            if interface not in self._factories:
                raise ValueError(
                    f"Service '{interface.__name__}' is not registered in the container."
                )

            instance = self._factories[interface]()

            if self._lifetimes.get(interface) == Lifetime.SINGLETON:
                self._singletons[interface] = instance

            return instance

    def is_registered(self, interface: Type[Any]) -> bool:
        """Check if a type is registered in the container."""
        with self._lock:
            return interface in self._factories or interface in self._singletons

    def reset(self) -> None:
        """Reset the container, removing all registrations."""
        with self._lock:
            self._factories.clear()
            self._singletons.clear()
            self._lifetimes.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Return the global DI container."""
    return _container


def inject(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that auto-injects dependencies from the global DI container.

    Inspects the function's type hints and resolves any parameter whose type
    is registered in the container, unless the caller supplies it explicitly.

    Example::

        @inject
        def my_handler(request: Request, service: MyService) -> Response:
            ...

    If ``MyService`` is registered in the container, it will be resolved
    automatically when ``my_handler`` is called without a ``service`` argument.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind_partial(*args, **kwargs)
        for param_name, param_type in hints.items():
            if param_name == "return":
                continue
            if param_name in bound.arguments:
                continue
            if _container.is_registered(param_type):
                kwargs[param_name] = _container.resolve(param_type)
        return func(*args, **kwargs)

    return wrapper

