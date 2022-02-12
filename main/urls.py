from  django.urls import path

from . import views

urlpatterns = [ 
    path("" , views.index , name = "index"),
    path("after_male", views.after_male, name = "after_male"),
    path("f_show_image_option" , views.f_show_image_option, name = "f_show_image_option"),
    path("m_show_image_option" , views.m_show_image_option, name = "m_show_image_option"),
    path("male_set2", views.male_set2, name="male_set2"),
    path("image_morph", views.image_morph, name = "image_morph"),
    path("edit", views.edit, name = "edit"),
    path("morphine" , views.morphine , name = "morphine"),
    path("image" , views.image , name = "image")
]