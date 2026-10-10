"""Общее для проверок: проект студента в пути импорта и параметры командной строки."""
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())  # запускайте pytest из корня своего проекта: там ваши модули и data/


def pytest_addoption(parser):
    parser.addoption("--db", default="support.db", help="база сервиса (уроки 04 и 11)")
    parser.addoption("--user", default=None, help="user_id, который попросил его забыть (урок 11)")
    parser.addoption("--canary", default=None, help="строка, которую этот пользователь написал в диалоге (урок 11)")


@pytest.fixture
def opt(request):
    return request.config.getoption
