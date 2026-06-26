// Removes enableBundleCompression from the generated android/app/build.gradle.
// Property was removed from ReactExtension in RN 0.76 but Expo SDK 54's
// prebuild template still sets it, causing a Gradle build failure.
const { withDangerousMod } = require('@expo/config-plugins');
const path = require('path');
const fs = require('fs');

module.exports = function withFixBundleCompression(config) {
  return withDangerousMod(config, [
    'android',
    (config) => {
      const gradlePath = path.join(
        config.modRequest.platformProjectRoot,
        'app',
        'build.gradle'
      );
      if (!fs.existsSync(gradlePath)) return config;

      let contents = fs.readFileSync(gradlePath, 'utf8');
      const patched = contents.replace(/^[^\n]*enableBundleCompression[^\n]*\n?/gm, '');

      if (patched !== contents) {
        fs.writeFileSync(gradlePath, patched, 'utf8');
        console.log('[fix-bundle-compression] Removed enableBundleCompression from app/build.gradle');
      }

      return config;
    },
  ]);
};
