import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, Dimensions, StatusBar, Platform,
  Animated, Easing,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SERVER, API_KEY } from './config';

// ── Iron Man palette ──────────────────────────────────────────────────────────
const RED    = '#FF3B30';
const GOLD   = '#FFD700';
const CYAN   = '#00E5FF';
const DIM    = 'rgba(255,215,0,0.25)';

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
  flat:      'PRESSURE STATUS:  ██░░░░░░░░  CRITICAL',
  defective: 'PRESSURE STATUS:  █████░░░░░  COMPROMISED',
  worn:      'PRESSURE STATUS:  ███████░░░  MONITOR',
  good:      'PRESSURE STATUS:  ██████████  NOMINAL',
  new:       'PRESSURE STATUS:  ██████████  OPTIMAL',
};

const { width: W, height: H } = Dimensions.get('window');

// Scan box region
const BOX = { x: W * 0.07, y: H * 0.17, w: W * 0.86, h: H * 0.50 };

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [result, setResult]   = useState(null);
  const [online, setOnline]   = useState(false);
  const [status, setStatus]   = useState('SCANNING');
  const [debug, setDebug]     = useState('');
  const cameraRef = useRef(null);
  const busy      = useRef(false);

  // Animations
  const scanAnim   = useRef(new Animated.Value(0)).current;
  const pulseAnim  = useRef(new Animated.Value(0)).current;
  const lockAnim   = useRef(new Animated.Value(0)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const flashAnim  = useRef(new Animated.Value(0)).current;

  const hasTire = result?.has_tire !== false && result?.class && result.class !== 'no_tire';
  const color   = hasTire ? (COND_COLOR[result.class] ?? GOLD) : GOLD;

  useEffect(() => {
    requestPermission();

    // Continuous scan sweep
    Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim, { toValue: 1, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(scanAnim, { toValue: 0, duration: 1400, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    ).start();

    // Pulse glow
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1, duration: 1000, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(pulseAnim, { toValue: 0, duration: 1000, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    ).start();

    // Slow rotation for HUD ring
    Animated.loop(
      Animated.timing(rotateAnim, { toValue: 1, duration: 8000, easing: Easing.linear, useNativeDriver: true })
    ).start();

    const timer = setInterval(capture, 700);
    return () => { clearInterval(timer); };
  }, []);

  // Flash + lock on tire detection
  useEffect(() => {
    if (hasTire) {
      setStatus('TARGET LOCKED');
      Animated.sequence([
        Animated.timing(flashAnim, { toValue: 1, duration: 120, useNativeDriver: false }),
        Animated.timing(flashAnim, { toValue: 0, duration: 300, useNativeDriver: false }),
      ]).start();
      Animated.timing(lockAnim, { toValue: 1, duration: 250, useNativeDriver: false }).start();
    } else {
      setStatus('SCANNING');
      Animated.timing(lockAnim, { toValue: 0, duration: 400, useNativeDriver: false }).start();
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
      setDebug(`${e.message}`);
    } finally {
      busy.current = false;
    }
  };

  // Derived animated values
  const scanLineY = scanAnim.interpolate({ inputRange: [0,1], outputRange: [BOX.y + 2, BOX.y + BOX.h - 4] });
  const glowOpacity = pulseAnim.interpolate({ inputRange: [0,1], outputRange: [0.4, 1.0] });
  const borderWidth = lockAnim.interpolate({ inputRange: [0,1], outputRange: [1.5, 3] });
  const flashOpacity = flashAnim.interpolate({ inputRange: [0,1], outputRange: [0, 0.6] });
  const cornerInset  = lockAnim.interpolate({ inputRange: [0,1], outputRange: [0, 6] });
  const spin = rotateAnim.interpolate({ inputRange: [0,1], outputRange: ['0deg','360deg'] });

  if (!permission?.granted) {
    return (
      <View style={s.root}>
        <Text style={s.permBtn} onPress={requestPermission}>ALLOW CAMERA ACCESS</Text>
      </View>
    );
  }

  return (
    <View style={s.root}>
      <StatusBar hidden />
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

      {/* Flash on lock */}
      <Animated.View style={[StyleSheet.absoluteFill, { backgroundColor: color, opacity: flashOpacity, zIndex: 5 }]} pointerEvents="none" />

      {/* Overlay layer */}
      <View style={StyleSheet.absoluteFill} pointerEvents="none">

        {/* Dim outside scan box */}
        <View style={[s.dim, { height: BOX.y }]} />
        <View style={{ flexDirection: 'row', height: BOX.h }}>
          <View style={[s.dim, { width: BOX.x }]} />
          <View style={{ flex: 1 }} />
          <View style={[s.dim, { width: BOX.x }]} />
        </View>
        <View style={[s.dim, { flex: 1 }]} />

        {/* Rotating HUD ring */}
        <Animated.View style={[s.hudRing, {
          left: W/2 - BOX.h*0.55, top: BOX.y + BOX.h/2 - BOX.h*0.55,
          width: BOX.h*1.1, height: BOX.h*1.1, borderRadius: BOX.h*0.55,
          transform: [{ rotate: spin }],
          borderColor: DIM,
        }]} />

        {/* Main scanning box border */}
        <Animated.View style={[s.boxBorder, {
          left: BOX.x, top: BOX.y, width: BOX.w, height: BOX.h,
          borderColor: hasTire ? color + 'aa' : GOLD + '33',
          borderWidth: borderWidth,
        }]} />

        {/* Corner brackets — Iron Man style */}
        {[
          { left: BOX.x,            top: BOX.y,              tl: true  },
          { left: BOX.x+BOX.w-36,  top: BOX.y,              tr: true  },
          { left: BOX.x,            top: BOX.y+BOX.h-36,    bl: true  },
          { left: BOX.x+BOX.w-36,  top: BOX.y+BOX.h-36,    br: true  },
        ].map((c, i) => (
          <Animated.View key={i} style={[s.corner, {
            left: c.left, top: c.top,
            borderColor: hasTire ? color : GOLD,
            borderTopWidth:    (c.tl || c.tr) ? 3 : 0,
            borderBottomWidth: (c.bl || c.br) ? 3 : 0,
            borderLeftWidth:   (c.tl || c.bl) ? 3 : 0,
            borderRightWidth:  (c.tr || c.br) ? 3 : 0,
            shadowColor: hasTire ? color : GOLD,
            shadowOpacity: 0.9, shadowRadius: 8, elevation: 8,
          }]} />
        ))}

        {/* Corner pip dots */}
        {[
          { left: BOX.x - 4,        top: BOX.y - 4       },
          { left: BOX.x+BOX.w - 4,  top: BOX.y - 4       },
          { left: BOX.x - 4,        top: BOX.y+BOX.h - 4 },
          { left: BOX.x+BOX.w - 4,  top: BOX.y+BOX.h - 4 },
        ].map((p, i) => (
          <Animated.View key={i} style={[s.pip, {
            left: p.left, top: p.top,
            backgroundColor: hasTire ? color : GOLD,
            opacity: glowOpacity,
          }]} />
        ))}

        {/* Scan line */}
        <Animated.View style={[s.scanLine, {
          left: BOX.x + 12, width: BOX.w - 24,
          top: scanLineY,
          backgroundColor: hasTire ? color : GOLD,
          shadowColor: hasTire ? color : GOLD,
          opacity: hasTire ? glowOpacity : 0.7,
        }]} />

        {/* Scan line leading glow */}
        <Animated.View style={[s.scanGlow, {
          left: BOX.x + 12, width: BOX.w - 24,
          top: scanLineY,
          backgroundColor: hasTire ? color : GOLD,
        }]} />

        {/* Crosshair center lines (subtle) */}
        <View style={[s.crossH, { top: BOX.y + BOX.h/2, left: BOX.x, width: BOX.w }]} />
        <View style={[s.crossV, { left: BOX.x + BOX.w/2, top: BOX.y, height: BOX.h }]} />

        {/* Status text inside box */}
        <View style={[s.statusInBox, { top: BOX.y + 10, left: BOX.x + 14 }]}>
          <Animated.Text style={[s.statusLabel, { color: hasTire ? color : GOLD, opacity: glowOpacity }]}>
            {status}
          </Animated.Text>
        </View>

        {/* Bottom-right confidence in box */}
        {hasTire && (
          <View style={[s.confInBox, { bottom: H - (BOX.y + BOX.h) + 10, right: W - (BOX.x + BOX.w) + 10 }]}>
            <Text style={[s.confInBoxTxt, { color }]}>{result.confidence}%</Text>
          </View>
        )}

      </View>

      {/* ── Header ── */}
      <View style={s.header}>
        <View>
          <Text style={s.headerTitle}>TIRE ANALYSIS SYSTEM</Text>
          <Text style={s.headerSub}>STARK INDUSTRIES  ·  v2.0</Text>
        </View>
        <View style={s.statusPill}>
          <View style={[s.onlineDot, { backgroundColor: online ? '#00FF88' : RED }]} />
          <Text style={[s.onlineTxt, { color: online ? '#00FF88' : RED }]}>
            {online ? 'ONLINE' : 'OFFLINE'}
          </Text>
        </View>
      </View>

      {/* ── Result HUD panel ── */}
      <View style={s.hud}>
        {hasTire ? (
          <>
            <View style={[s.hudDivider, { backgroundColor: color }]} />
            <View style={s.hudRow}>
              <Text style={s.hudKey}>CONDITION</Text>
              <Text style={[s.hudVal, { color }]}>{COND_LABEL[result.class] ?? result.label}</Text>
            </View>
            <View style={s.confBarRow}>
              <Text style={s.hudKey}>CERTAINTY</Text>
              <View style={s.confTrack}>
                <Animated.View style={[s.confFill, { width: `${result.confidence}%`, backgroundColor: color }]} />
              </View>
              <Text style={[s.pctTxt, { color }]}>{result.confidence}%</Text>
            </View>
            {PRESSURE[result.class] ? (
              <Text style={[s.pressureTxt, { color: color + 'cc' }]}>{PRESSURE[result.class]}</Text>
            ) : null}
            <View style={[s.hudDivider, { backgroundColor: color + '44', marginTop: 10 }]} />
          </>
        ) : (
          <View style={s.idleHud}>
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
  root:   { flex: 1, backgroundColor: '#000' },

  dim:    { backgroundColor: 'rgba(0,0,0,0.55)' },

  hudRing: {
    position: 'absolute',
    borderWidth: 1,
    borderStyle: 'dashed',
  },

  boxBorder: { position: 'absolute' },

  corner: { position: 'absolute', width: 36, height: 36 },

  pip: {
    position: 'absolute', width: 7, height: 7, borderRadius: 4,
    shadowOpacity: 1, shadowRadius: 6, elevation: 6,
  },

  scanLine: {
    position: 'absolute', height: 2,
    shadowOpacity: 1, shadowRadius: 10, elevation: 10,
  },

  scanGlow: {
    position: 'absolute', height: 18,
    opacity: 0.12,
    marginTop: -8,
    shadowOpacity: 0.6, shadowRadius: 20,
  },

  crossH: {
    position: 'absolute', height: 1,
    backgroundColor: 'rgba(255,215,0,0.08)',
  },
  crossV: {
    position: 'absolute', width: 1,
    backgroundColor: 'rgba(255,215,0,0.08)',
  },

  statusInBox: { position: 'absolute' },
  statusLabel: { fontSize: 11, fontWeight: '700', letterSpacing: 3 },

  confInBox:    { position: 'absolute' },
  confInBoxTxt: { fontSize: 13, fontWeight: '800', letterSpacing: 1 },

  // Header
  header: {
    position: 'absolute', top: 0, left: 0, right: 0,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end',
    paddingTop: Platform.OS === 'ios' ? 52 : 36,
    paddingHorizontal: 20, paddingBottom: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderBottomWidth: 1, borderBottomColor: GOLD + '33',
  },
  headerTitle: { color: GOLD, fontSize: 13, fontWeight: '800', letterSpacing: 4 },
  headerSub:   { color: GOLD + '66', fontSize: 9, letterSpacing: 2, marginTop: 2 },

  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  onlineDot:  { width: 6, height: 6, borderRadius: 3 },
  onlineTxt:  { fontSize: 10, fontWeight: '700', letterSpacing: 2 },

  // HUD panel
  hud: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    paddingBottom: Platform.OS === 'ios' ? 40 : 24,
    paddingHorizontal: 20, paddingTop: 14,
    backgroundColor: 'rgba(0,0,0,0.82)',
    borderTopWidth: 1, borderTopColor: GOLD + '44',
  },

  hudDivider: { height: 1, marginBottom: 10 },

  hudRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  hudKey: { color: '#4B5563', fontSize: 10, fontWeight: '700', letterSpacing: 2, width: 90 },
  hudVal: { fontSize: 14, fontWeight: '800', letterSpacing: 1, flex: 1 },

  confBarRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  confTrack:  { flex: 1, height: 4, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden', marginHorizontal: 10 },
  confFill:   { height: '100%', borderRadius: 2 },
  pctTxt:     { fontSize: 12, fontWeight: '700', width: 40, textAlign: 'right' },

  pressureTxt: { fontSize: 11, letterSpacing: 1, fontWeight: '600', marginTop: 4 },

  idleHud: { alignItems: 'center', paddingVertical: 8 },
  idleTxt: { color: GOLD, fontSize: 12, fontWeight: '700', letterSpacing: 4 },
  debugTxt: { color: '#4B5563', fontSize: 10, marginTop: 4, letterSpacing: 1 },

  permBtn: { color: GOLD, fontSize: 16, fontWeight: '700', letterSpacing: 3,
    position: 'absolute', top: '50%', alignSelf: 'center' },
});
