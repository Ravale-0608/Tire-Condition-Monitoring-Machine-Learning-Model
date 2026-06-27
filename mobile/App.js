import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  View, Text, StyleSheet, Dimensions, StatusBar, Platform,
  Animated, Easing,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SERVER, API_KEY } from './config';

// ── Iron Man palette ──────────────────────────────────────────────────────────
const GOLD   = '#FFD700';
const DIM    = 'rgba(255,215,0,0.20)';

const COND_COLOR = {
  no_tire:   '#4B5563',
  flat:      '#FF3B30',
  defective: '#FF6B35',
  worn:      '#FFD700',
  good:      '#00FF88',
  new:       '#00BFFF',
};

const COND_LABEL = {
  no_tire:   'NO TARGET',
  flat:      'FLAT TIRE — CRITICAL',
  defective: 'DEFECTIVE — DANGER',
  worn:      'WORN TREAD — WARNING',
  good:      'CONDITION NOMINAL',
  new:       'NEW TIRE — OPTIMAL',
};

const PRESSURE = {
  no_tire:   '',
  flat:      'PRESSURE  ██░░░░░░░░  CRITICAL',
  defective: 'PRESSURE  █████░░░░░  COMPROMISED',
  worn:      'PRESSURE  ███████░░░  MONITOR',
  good:      'PRESSURE  ██████████  NOMINAL',
  new:       'PRESSURE  ██████████  OPTIMAL',
};

// Grid config
const COLS = 9;
const ROWS = 7;

const { width: W, height: H } = Dimensions.get('window');
const BOX  = { x: W * 0.07, y: H * 0.17, w: W * 0.86, h: H * 0.50 };
const CELL = { w: BOX.w / COLS, h: BOX.h / ROWS };

// Pre-compute cell "intensity" weights — brighter near center (heat-map feel)
const cellWeights = Array.from({ length: ROWS }, (_, r) =>
  Array.from({ length: COLS }, (_, c) => {
    const dr = (r - ROWS / 2) / (ROWS / 2);
    const dc = (c - COLS / 2) / (COLS / 2);
    return Math.max(0, 1 - Math.sqrt(dr * dr + dc * dc));
  })
);

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [result, setResult]   = useState(null);
  const [online, setOnline]   = useState(false);
  const [status, setStatus]   = useState('SCANNING');
  const [debug, setDebug]     = useState('');
  const cameraRef = useRef(null);
  const busy      = useRef(false);

  const scanAnim    = useRef(new Animated.Value(0)).current;
  const pulseAnim   = useRef(new Animated.Value(0)).current;
  const rotateAnim  = useRef(new Animated.Value(0)).current;
  const flashAnim   = useRef(new Animated.Value(0)).current;
  const heatAnim    = useRef(new Animated.Value(0)).current;  // grid heat reveal

  const hasTire = result?.has_tire !== false && result?.class && result.class !== 'no_tire';
  const color   = hasTire ? (COND_COLOR[result.class] ?? GOLD) : GOLD;

  useEffect(() => {
    requestPermission();

    Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim,  { toValue: 1, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(scanAnim,  { toValue: 0, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    ).start();

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1, duration: 900,  easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(pulseAnim, { toValue: 0, duration: 900,  easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    ).start();

    Animated.loop(
      Animated.timing(rotateAnim, { toValue: 1, duration: 10000, easing: Easing.linear, useNativeDriver: true })
    ).start();

    const timer = setInterval(capture, 700);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (hasTire) {
      setStatus('TARGET LOCKED');
      Animated.sequence([
        Animated.timing(flashAnim, { toValue: 1, duration: 100, useNativeDriver: false }),
        Animated.timing(flashAnim, { toValue: 0, duration: 250, useNativeDriver: false }),
      ]).start();
      Animated.timing(heatAnim, { toValue: 1, duration: 600, easing: Easing.out(Easing.quad), useNativeDriver: false }).start();
    } else {
      setStatus('SCANNING');
      Animated.timing(heatAnim, { toValue: 0, duration: 400, useNativeDriver: false }).start();
    }
  }, [hasTire]);

  const capture = async () => {
    if (busy.current || !cameraRef.current) return;
    busy.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
      const form  = new FormData();
      form.append('file', { uri: photo.uri, type: 'image/jpeg', name: 'frame.jpg' });
      const res   = await fetch(`${SERVER}/predict?key=${API_KEY}`, { method: 'POST', body: form });
      const data  = await res.json();
      setOnline(true);
      setResult(data.error ? null : data);
      setDebug(data.error ? `ERR: ${data.error}` : `${data.class ?? 'no_tire'} ${data.confidence ?? 0}%`);
    } catch (e) {
      setOnline(false);
      setResult(null);
      setDebug(e.message ?? 'error');
    } finally {
      busy.current = false;
    }
  };

  // Derived values
  const scanLineY    = scanAnim.interpolate({ inputRange: [0, 1], outputRange: [BOX.y + 2, BOX.y + BOX.h - 4] });
  const glowOpacity  = pulseAnim.interpolate({ inputRange: [0, 1], outputRange: [0.4, 1.0] });
  const flashOpacity = flashAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 0.55] });
  const heatOpacity  = heatAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 1] });
  const spin         = rotateAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  if (!permission?.granted) {
    return (
      <View style={s.root}>
        <Text style={[s.permBtn]} onPress={requestPermission}>ALLOW CAMERA ACCESS</Text>
      </View>
    );
  }

  return (
    <View style={s.root}>
      <StatusBar hidden />
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

      {/* Lock-on flash */}
      <Animated.View pointerEvents="none"
        style={[StyleSheet.absoluteFill, { backgroundColor: color, opacity: flashOpacity, zIndex: 5 }]} />

      <View style={StyleSheet.absoluteFill} pointerEvents="none">

        {/* Dim vignette outside box */}
        <View style={[s.dim, { height: BOX.y }]} />
        <View style={{ flexDirection: 'row', height: BOX.h }}>
          <View style={[s.dim, { width: BOX.x }]} />
          <View style={{ flex: 1 }} />
          <View style={[s.dim, { width: BOX.x }]} />
        </View>
        <View style={[s.dim, { flex: 1 }]} />

        {/* ── SCAN GRID ─────────────────────────────────────── */}

        {/* Horizontal grid lines */}
        {Array.from({ length: ROWS + 1 }).map((_, i) => (
          <View key={`h${i}`} style={{
            position: 'absolute',
            left: BOX.x, top: BOX.y + CELL.h * i,
            width: BOX.w, height: 1,
            backgroundColor: hasTire ? color + '30' : 'rgba(255,215,0,0.13)',
          }} />
        ))}

        {/* Vertical grid lines */}
        {Array.from({ length: COLS + 1 }).map((_, i) => (
          <View key={`v${i}`} style={{
            position: 'absolute',
            top: BOX.y, left: BOX.x + CELL.w * i,
            width: 1, height: BOX.h,
            backgroundColor: hasTire ? color + '30' : 'rgba(255,215,0,0.13)',
          }} />
        ))}

        {/* ── HEAT MAP overlay on detection ─────────────────── */}
        {cellWeights.map((row, r) =>
          row.map((weight, c) => (
            <Animated.View key={`cell_${r}_${c}`} style={{
              position: 'absolute',
              left: BOX.x + CELL.w * c + 1,
              top:  BOX.y + CELL.h * r + 1,
              width:  CELL.w - 2,
              height: CELL.h - 2,
              backgroundColor: color,
              opacity: heatOpacity.interpolate({
                inputRange: [0, 1],
                outputRange: [0, weight * 0.28],
              }),
            }} />
          ))
        )}

        {/* ── SCAN GLOW BAND sweeping over the grid ─────────── */}
        <Animated.View style={{
          position: 'absolute',
          left: BOX.x, width: BOX.w,
          height: CELL.h * 1.8,
          top: Animated.subtract(scanLineY, CELL.h * 0.9),
          backgroundColor: hasTire ? color : GOLD,
          opacity: hasTire ? 0.08 : 0.06,
        }} />

        {/* Corner intersection dots on grid (data node feel) */}
        {Array.from({ length: ROWS + 1 }).map((_, r) =>
          Array.from({ length: COLS + 1 }).map((_, c) => (
            <Animated.View key={`dot_${r}_${c}`} style={{
              position: 'absolute',
              left: BOX.x + CELL.w * c - 2,
              top:  BOX.y + CELL.h * r - 2,
              width: 4, height: 4, borderRadius: 2,
              backgroundColor: hasTire ? color : GOLD,
              opacity: glowOpacity.interpolate({
                inputRange: [0, 1],
                outputRange: [0.15, hasTire ? 0.55 : 0.25],
              }),
            }} />
          ))
        )}

        {/* ── SCAN LINE ─────────────────────────────────────── */}
        {/* Glow bloom behind line */}
        <Animated.View style={{
          position: 'absolute',
          left: BOX.x + 8, width: BOX.w - 16,
          height: 20, top: Animated.subtract(scanLineY, 10),
          backgroundColor: hasTire ? color : GOLD,
          opacity: 0.10,
        }} />
        {/* Main scan line */}
        <Animated.View style={{
          position: 'absolute',
          left: BOX.x + 8, width: BOX.w - 16,
          height: 2, top: scanLineY,
          backgroundColor: hasTire ? color : GOLD,
          shadowColor: hasTire ? color : GOLD,
          shadowOpacity: 1, shadowRadius: 8, elevation: 10,
          opacity: glowOpacity,
        }} />
        {/* Bright core of scan line */}
        <Animated.View style={{
          position: 'absolute',
          left: BOX.x + BOX.w * 0.3, width: BOX.w * 0.4,
          height: 1, top: scanLineY,
          backgroundColor: '#fff',
          opacity: glowOpacity.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.9] }),
        }} />

        {/* ── ROTATING HUD RING (background decoration) ─────── */}
        <Animated.View style={[s.hudRing, {
          left: W / 2 - BOX.h * 0.52, top: BOX.y + BOX.h / 2 - BOX.h * 0.52,
          width: BOX.h * 1.04, height: BOX.h * 1.04, borderRadius: BOX.h * 0.52,
          borderColor: DIM, transform: [{ rotate: spin }],
        }]} />
        <Animated.View style={[s.hudRing, {
          left: W / 2 - BOX.h * 0.44, top: BOX.y + BOX.h / 2 - BOX.h * 0.44,
          width: BOX.h * 0.88, height: BOX.h * 0.88, borderRadius: BOX.h * 0.44,
          borderColor: DIM, transform: [{ rotate: spin }], borderStyle: 'dashed',
          opacity: 0.5,
        }]} />

        {/* ── CORNER BRACKETS ───────────────────────────────── */}
        {[
          { l: BOX.x,            t: BOX.y,            bT: 3, bL: 3, bR: 0, bB: 0 },
          { l: BOX.x+BOX.w-36,  t: BOX.y,            bT: 3, bR: 3, bL: 0, bB: 0 },
          { l: BOX.x,            t: BOX.y+BOX.h-36,  bB: 3, bL: 3, bT: 0, bR: 0 },
          { l: BOX.x+BOX.w-36,  t: BOX.y+BOX.h-36,  bB: 3, bR: 3, bT: 0, bL: 0 },
        ].map(({ l, t, bT, bB, bL, bR }, i) => (
          <Animated.View key={i} style={{
            position: 'absolute', left: l, top: t, width: 36, height: 36,
            borderTopWidth: bT, borderBottomWidth: bB, borderLeftWidth: bL, borderRightWidth: bR,
            borderColor: hasTire ? color : GOLD,
            shadowColor: hasTire ? color : GOLD,
            shadowOpacity: 0.9, shadowRadius: 10, elevation: 10,
          }} />
        ))}

        {/* ── STATUS TEXT inside box ────────────────────────── */}
        <View style={{ position: 'absolute', top: BOX.y + 10, left: BOX.x + 14 }}>
          <Animated.Text style={[s.statusLabel, { color: hasTire ? color : GOLD, opacity: glowOpacity }]}>
            {status}
          </Animated.Text>
        </View>

        {/* Confidence inside box (top right) */}
        {hasTire && (
          <View style={{ position: 'absolute', top: BOX.y + 10, right: W - (BOX.x + BOX.w) + 14 }}>
            <Text style={[s.confInBox, { color }]}>{result.confidence}%</Text>
          </View>
        )}

      </View>

      {/* ── HEADER ── */}
      <View style={s.header}>
        <View>
          <Text style={s.headerTitle}>TIRE ANALYSIS SYSTEM</Text>
          <Text style={s.headerSub}>STARK INDUSTRIES  ·  v2.0</Text>
        </View>
        <View style={s.statusPill}>
          <View style={[s.dot, { backgroundColor: online ? '#00FF88' : '#FF3B30' }]} />
          <Text style={[s.dotTxt, { color: online ? '#00FF88' : '#FF3B30' }]}>
            {online ? 'ONLINE' : 'OFFLINE'}
          </Text>
        </View>
      </View>

      {/* ── HUD RESULT PANEL ── */}
      <View style={s.hud}>
        {hasTire ? (
          <>
            <View style={[s.divider, { backgroundColor: color }]} />
            <View style={s.row}>
              <Text style={s.hudKey}>CONDITION</Text>
              <Text style={[s.hudVal, { color }]}>{COND_LABEL[result.class] ?? result.label}</Text>
            </View>
            <View style={s.row}>
              <Text style={s.hudKey}>CERTAINTY</Text>
              <View style={s.confTrack}>
                <View style={[s.confFill, { width: `${result.confidence}%`, backgroundColor: color }]} />
              </View>
              <Text style={[s.pct, { color }]}>{result.confidence}%</Text>
            </View>
            {PRESSURE[result.class] ? (
              <Text style={[s.pressureTxt, { color: color + 'bb' }]}>{PRESSURE[result.class]}</Text>
            ) : null}
            <View style={[s.divider, { backgroundColor: color + '33', marginTop: 10 }]} />
          </>
        ) : (
          <View style={s.idle}>
            <Animated.Text style={[s.idleTxt, { opacity: glowOpacity }]}>
              POINT CAMERA AT TIRE
            </Animated.Text>
            <Text style={s.debugTxt}>{debug}</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  root:       { flex: 1, backgroundColor: '#000' },
  dim:        { backgroundColor: 'rgba(0,0,0,0.55)' },

  hudRing:    { position: 'absolute', borderWidth: 1 },

  statusLabel:{ fontSize: 11, fontWeight: '800', letterSpacing: 3 },
  confInBox:  { fontSize: 13, fontWeight: '800', letterSpacing: 1 },

  header: {
    position: 'absolute', top: 0, left: 0, right: 0,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end',
    paddingTop: Platform.OS === 'ios' ? 52 : 36,
    paddingHorizontal: 20, paddingBottom: 10,
    backgroundColor: 'rgba(0,0,0,0.72)',
    borderBottomWidth: 1, borderBottomColor: 'rgba(255,215,0,0.25)',
  },
  headerTitle:{ color: GOLD, fontSize: 13, fontWeight: '800', letterSpacing: 4 },
  headerSub:  { color: 'rgba(255,215,0,0.5)', fontSize: 9, letterSpacing: 2, marginTop: 2 },
  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dot:        { width: 6, height: 6, borderRadius: 3 },
  dotTxt:     { fontSize: 10, fontWeight: '700', letterSpacing: 2 },

  hud: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    paddingBottom: Platform.OS === 'ios' ? 40 : 24,
    paddingHorizontal: 20, paddingTop: 14,
    backgroundColor: 'rgba(0,0,0,0.82)',
    borderTopWidth: 1, borderTopColor: 'rgba(255,215,0,0.3)',
  },
  divider:    { height: 1, marginBottom: 10 },
  row:        { flexDirection: 'row', alignItems: 'center', marginBottom: 7 },
  hudKey:     { color: '#4B5563', fontSize: 10, fontWeight: '700', letterSpacing: 2, width: 90 },
  hudVal:     { fontSize: 14, fontWeight: '800', letterSpacing: 1, flex: 1 },
  confTrack:  { flex: 1, height: 4, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden', marginHorizontal: 10 },
  confFill:   { height: '100%', borderRadius: 2 },
  pct:        { fontSize: 12, fontWeight: '700', width: 40, textAlign: 'right' },
  pressureTxt:{ fontSize: 11, letterSpacing: 1, fontWeight: '600', marginTop: 2 },

  idle:       { alignItems: 'center', paddingVertical: 8 },
  idleTxt:    { color: GOLD, fontSize: 12, fontWeight: '700', letterSpacing: 4 },
  debugTxt:   { color: '#4B5563', fontSize: 10, marginTop: 4, letterSpacing: 1 },

  permBtn:    { color: GOLD, fontSize: 16, fontWeight: '700', letterSpacing: 3,
                position: 'absolute', top: '50%', alignSelf: 'center' },
});
