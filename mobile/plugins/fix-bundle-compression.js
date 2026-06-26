// Removes enableBundleCompression from the generated app/build.gradle.
// This property was deleted from ReactExtension in React Native 0.76
// but Expo SDK 54's prebuild template still sets it, causing a Gradle failure.
const { withAppBuildGradle } = require('@expo/config-plugins');

module.exports = withAppBuildGradle((config) => {
  config.modResults.contents = config.modResults.contents.replace(
    /\s*enableBundleCompression\s*=\s*(true|false)\n?/g,
    '\n'
  );
  return config;
});
