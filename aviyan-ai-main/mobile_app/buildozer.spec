[app]

# --------------------------------------------------
# AVIYAN AI - Android Application
# AB DEV STUDIO
# Founder & Creator: ABISHEK BHUSAL
# --------------------------------------------------

# Application name
title = AVIYAN AI

# Package information
package.name = aviyan
package.domain = com.abdevstudio

# Source directory
source.dir = .

# Files included in APK
source.include_exts = py,png,jpg,jpeg,kv,atlas,json,txt

# Application version
version = 1.1.0

# Python/Kivy dependencies
requirements = python3,kivy

# Screen orientation
orientation = portrait

# Do not force fullscreen
fullscreen = 0

# --------------------------------------------------
# Android configuration
# --------------------------------------------------

# Android API
android.api = 35

# Minimum supported Android API
android.minapi = 24

# Android NDK
android.ndk = 28c

# Explicit Android Build Tools version
android.build_tools_version = 35.0.0

# Automatically accept Android SDK licenses
android.accept_sdk_license = True

# --------------------------------------------------
# Android permissions
# --------------------------------------------------

android.permissions = INTERNET

# --------------------------------------------------
# Android architecture
# --------------------------------------------------

android.archs = arm64-v8a

# --------------------------------------------------
# Android application settings
# --------------------------------------------------

android.allow_backup = True

# --------------------------------------------------
# Build settings
# --------------------------------------------------

# Do not use deprecated Android ANT
android.skip_update = False

# --------------------------------------------------
# Logging
# --------------------------------------------------

log_level = 2

# --------------------------------------------------
# Warn when Buildozer runs as root
# --------------------------------------------------

warn_on_root = 1