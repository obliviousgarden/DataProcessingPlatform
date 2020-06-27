from django.urls import path
from .tensorflow import tem

urlpatterns = [
    path(r'tf/tem', tem.tf_tem),

]
