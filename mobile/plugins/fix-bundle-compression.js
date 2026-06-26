// Removes enableBundleCompression from the generated app/build.gradle.
// This property was deleted from ReactExtension in React Native 0.76
// but Expo SDK 54's prebuild template still sets it, causing a Gradle failure.
const { withAppBuildGradle } = require('@expo/config-plugins');

module.exports = withAppBuildGradle((config) => {
  if (!config.modResults?.contents) return config;
  config.modResults.contents = config.modResults.contents.replace(
    /[ \t]*enableBundleCompression\s*=\s*(true|false)\s*\n?/g,
    ''
  );
  return config;
});
