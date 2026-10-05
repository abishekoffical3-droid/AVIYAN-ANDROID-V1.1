[app]
# AVIYAN Android application
# Creator: ABISHEK BHUSAL | AB DEV STUDIO | NEPAL

package.name = aviyan
package.domain = app.aviyan
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,json,atlas,txt
version = 1.0.0
requirements = python3,kivy==2.3.1
orientation = portrait
fullscreen = 0
android.api = 35
android.minapi = 23
android.ndk = 27c
android.permissions = INTERNET,ACCESS_NETWORK_STATE
android.archs = arm64-v8a,armeabi-v7a
android.accept_sdk_license = True
android.allow_backup = True
android.presplash_color = #090B10

# Keep build deterministic and avoid bundling the large training/model tree.
# The Android app is the client; heavy model/training runs in the AVIYAN backend.

[buildozer]
log_level = 2
warn_on_root = 1
