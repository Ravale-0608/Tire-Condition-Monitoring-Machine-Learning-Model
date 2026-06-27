import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, Dimensions, StatusBar, Platform,
  Animated, Easing,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

import { SERVER } from './config';

const COLORS = {
  no_tire:   '#6b7280',
  flat:      '#ef4444',
  defective: '#f97316',
  worn:      '#eab308',
  good:      '#22c55e',
  new:       '#3b82f6',
};

const LABELS = {
  no_tire:   'No Tire Detected',
  flat:      '⚠️  Flat Tire',
  defective: '🔴  Defective',
  worn:      '🟡  Worn Tread',
  good:      '✅  Good Condition',
  new:       '🔵  New Tire',
};

const { width: W, height: H } = Dimensions.get('window');
const CORNER = 28;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [result, setResult]   = useState(null);
  const [online, setOnline]   = useState(false);
  const cameraRef  = useRef(null);
  const busy       = useRef(false);
  const scanAnim   = useRef(new Animated.Value(0)).current;

  const box = { x: W * 0.08, y: H * 0.18, w: W * 0.84, h: H * 0.50 };

  useEffect(() => {
    requestPermission();
    Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim, { toValue: 1, duration: 1800, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(scanAnim, { toValue: 0, duration: 1800, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    ).start();
    const timer = setInterval(capture, 700);
    return () => { clearInterval(timer); scanAnim.stopAnimation(); };
  }, []);

  const capture = async () => {
    if (busy.current || !cameraRef.current) return;
    busy.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.25, skipProcessing: true,
      });
      const form = new FormData();
      form.append('file', { uri: photo.uri, type: 'image/jpeg', name: 'frame.jpg' });
      const res  = await fetch(`${SERVER}/predict`, { method: 'POST', body: form });
      const data = await res.json();
      setOnline(true);
      setResult(data.error ? null : data);
    } catch {
      setOnline(false);
      setResult(null);
    } finally {
      busy.current = false;
    }
  };

  const hasTire = result?.has_tire !== false && result?.class && result.class !== 'no_tire';
  const color   = hasTire ? (COLORS[result.class] ?? '#22c55e') : '#3b82f6';

  const lineTop = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [box.y, box.y + box.h - 2],
  });

  if (!permission) return <View style={s.root} />;

  if (!permission.granted) {
    return (
      <View style={s.center}>
        <Text style={s.permBtn} onPress={requestPermission}>Allow Camera Access</Text>
      </View>
    );
  }

  return (
    <View style={s.root}>
      <StatusBar hidden />
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

      {/* Overlay */}
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {/* Dim outside scan box */}
        <View style={[s.dim, { height: box.y }]} />
        <View style={{ flexDirection: 'row', height: box.h }}>
          <View style={[s.dim, { width: box.x }]} />
          <View style={{ width: box.w }} />
          <View style={[s.dim, { flex: 1 }]} />
        </View>
        <View style={[s.dim, { flex: 1 }]} />

        {/* Box border */}
        <View style={[s.boxBorder, {
          left: box.x, top: box.y, width: box.w, height: box.h,
          borderColor: color + '44',
        }]} />

        {/* Corner brackets */}
        {[
          { left: box.x,              top: box.y,               borderTopWidth: 3, borderLeftWidth: 3 },
          { left: box.x+box.w-CORNER, top: box.y,               borderTopWidth: 3, borderRightWidth: 3 },
          { left: box.x,              top: box.y+box.h-CORNER,  borderBottomWidth: 3, borderLeftWidth: 3 },
          { left: box.x+box.w-CORNER, top: box.y+box.h-CORNER,  borderBottomWidth: 3, borderRightWidth: 3 },
        ].map((style, i) => (
          <View key={i} style={[s.corner, { borderColor: color }, style]} />
        ))}

        {/* Animated scan line */}
        <Animated.View style={[
          s.scanLine,
          { left: box.x + 8, width: box.w - 16, backgroundColor: color, top: lineTop },
        ]} />
      </View>

      {/* Header */}
      <View style={s.header}>
        <Text style={s.title}>🔍 Tire Scanner</Text>
        <View style={[s.dot, { backgroundColor: online ? '#22c55e' : '#ef4444' }]} />
      </View>

      {/* Result bar */}
      <View style={s.resultArea}>
        {hasTire ? (
          <>
            <Text style={[s.conditionText, { color }]}>
              {LABELS[result.class] ?? result.label}
            </Text>
            <View style={s.confTrack}>
              <View style={[s.confFill, { width: `${result.confidence}%`, backgroundColor: color }]} />
            </View>
            <Text style={s.confLabel}>{result.confidence}% confidence</Text>
          </>
        ) : (
          <Text style={s.hint}>
            {online ? 'Point camera at a tire' : 'Connecting to scanner…'}
          </Text>
        )}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  root:   { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000' },
  permBtn:{ color: '#3b82f6', fontSize: 17, fontWeight: '600' },

  dim: { backgroundColor: 'rgba(0,0,0,0.45)' },

  boxBorder: { position: 'absolute', borderWidth: 1 },
  corner:    { position: 'absolute', width: CORNER, height: CORNER },

  scanLine: {
    position: 'absolute', height: 2, opacity: 0.9,
    shadowColor: '#fff', shadowOpacity: 0.5,
    shadowRadius: 6, shadowOffset: { width: 0, height: 0 },
  },

  header: {
    position: 'absolute', top: 0, left: 0, right: 0,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 56 : 40,
    paddingHorizontal: 24, paddingBottom: 12,
  },
  title: { color: '#fff', fontSize: 18, fontWeight: '700' },
  dot:   { width: 9, height: 9, borderRadius: 5 },

  resultArea: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    paddingBottom: Platform.OS === 'ios' ? 48 : 32,
    paddingHorizontal: 24, paddingTop: 20,
    backgroundColor: 'rgba(0,0,0,0.72)',
  },
  conditionText: { fontSize: 26, fontWeight: '800', letterSpacing: -0.5 },
  confTrack: {
    height: 4, backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 2, marginTop: 10, overflow: 'hidden',
  },
  confFill:  { height: '100%', borderRadius: 2 },
  confLabel: { color: 'rgba(255,255,255,0.5)', fontSize: 12, marginTop: 5 },
  hint:      { color: 'rgba(255,255,255,0.45)', fontSize: 16 },
});
