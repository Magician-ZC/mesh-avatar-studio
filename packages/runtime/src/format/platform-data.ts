// Optional manifest blocks written by a hosting platform's build pipeline (WebGL runtime only;
// the native runtime ignores them). Both are plain data: the runtime does not act on them, it only
// validates and preserves them so that a platform can sign and enforce them.

/**
 * Final capability list produced after quality checks. The structure mirrors a product capability
 * list; a field that is absent was not granted. Unknown keys are ignored.
 */
export interface AvatarCapabilitiesData {
  style?: string;
  face?: { blink?: boolean; mouth?: boolean; gaze?: boolean; brows?: boolean };
  head?: { turn?: boolean; roll?: boolean };
  body?: { breath?: boolean; lean?: boolean; sway?: boolean };
  legs?: boolean;
  arms?: boolean;
  limbs?: { name: string; kind: string; free: boolean }[];
  strands?: string[];
  cloth?: string[];
  zones?: string[];
  motions?: { id: string; idle: boolean }[];
}

/** Ownership and provenance stamped by a hosting platform. All fields are required when present. */
export interface AvatarPlatformData {
  avatarId: string;
  ownerId: string;
  region: string;
  issuedAt: string;
  style: string;
  aiGenerated: boolean;
}

type Errors = string[];
const isObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$/;

function text(value: unknown, path: string, errors: Errors): string | undefined {
  if (typeof value === 'string' && value.trim()) return value;
  errors.push(`${path}: expected a non-empty string`);
}

function flag(value: unknown, path: string, errors: Errors): boolean | undefined {
  if (typeof value === 'boolean') return value;
  errors.push(`${path}: expected a boolean`);
}

function flags<K extends string>(value: unknown, keys: readonly K[], path: string, errors: Errors): Partial<Record<K, boolean>> | undefined {
  if (!isObject(value)) { errors.push(`${path}: expected an object`); return; }
  const result: Partial<Record<K, boolean>> = {};
  for (const key of keys) {
    if (value[key] === undefined) continue;
    const parsed = flag(value[key], `${path}.${key}`, errors);
    if (parsed !== undefined) result[key] = parsed;
  }
  return result;
}

function strings(value: unknown, path: string, errors: Errors): string[] | undefined {
  if (!Array.isArray(value)) { errors.push(`${path}: expected an array`); return; }
  const result: string[] = [];
  value.forEach((item, i) => { const parsed = text(item, `${path}[${i}]`, errors); if (parsed !== undefined) result.push(parsed); });
  return result;
}

function records<T>(value: unknown, path: string, errors: Errors, parse: (item: Record<string, unknown>, path: string) => T | undefined): T[] | undefined {
  if (!Array.isArray(value)) { errors.push(`${path}: expected an array`); return; }
  const result: T[] = [];
  value.forEach((item, i) => {
    if (!isObject(item)) { errors.push(`${path}[${i}]: expected an object`); return; }
    const parsed = parse(item, `${path}[${i}]`);
    if (parsed !== undefined) result.push(parsed);
  });
  return result;
}

/** Validate a `finalCapabilities` block, returning a new object with the known fields. */
export function parseCapabilitiesData(value: unknown, path: string, errors: Errors): AvatarCapabilitiesData | undefined {
  if (!isObject(value)) { errors.push(`${path}: expected an object`); return; }
  const result: AvatarCapabilitiesData = {};
  const before = errors.length;
  if (value.style !== undefined) result.style = text(value.style, `${path}.style`, errors);
  if (value.face !== undefined) result.face = flags(value.face, ['blink', 'mouth', 'gaze', 'brows'] as const, `${path}.face`, errors);
  if (value.head !== undefined) result.head = flags(value.head, ['turn', 'roll'] as const, `${path}.head`, errors);
  if (value.body !== undefined) result.body = flags(value.body, ['breath', 'lean', 'sway'] as const, `${path}.body`, errors);
  if (value.legs !== undefined) result.legs = flag(value.legs, `${path}.legs`, errors);
  if (value.arms !== undefined) result.arms = flag(value.arms, `${path}.arms`, errors);
  if (value.limbs !== undefined) result.limbs = records(value.limbs, `${path}.limbs`, errors, (item, at) => {
    const name = text(item.name, `${at}.name`, errors), kind = text(item.kind, `${at}.kind`, errors), free = flag(item.free, `${at}.free`, errors);
    return name !== undefined && kind !== undefined && free !== undefined ? { name, kind, free } : undefined;
  });
  for (const key of ['strands', 'cloth', 'zones'] as const) {
    if (value[key] !== undefined) result[key] = strings(value[key], `${path}.${key}`, errors);
  }
  if (value.motions !== undefined) result.motions = records(value.motions, `${path}.motions`, errors, (item, at) => {
    const id = text(item.id, `${at}.id`, errors), idle = flag(item.idle, `${at}.idle`, errors);
    return id !== undefined && idle !== undefined ? { id, idle } : undefined;
  });
  return errors.length === before ? result : undefined;
}

/** Validate a `platform` block, returning a new object with the known fields. */
export function parsePlatformData(value: unknown, path: string, errors: Errors): AvatarPlatformData | undefined {
  if (!isObject(value)) { errors.push(`${path}: expected an object`); return; }
  const before = errors.length;
  const avatarId = text(value.avatarId, `${path}.avatarId`, errors);
  const ownerId = text(value.ownerId, `${path}.ownerId`, errors);
  const region = text(value.region, `${path}.region`, errors);
  let issuedAt = text(value.issuedAt, `${path}.issuedAt`, errors);
  if (issuedAt !== undefined && (!ISO_TIMESTAMP.test(issuedAt) || Number.isNaN(Date.parse(issuedAt)))) {
    errors.push(`${path}.issuedAt: expected an ISO 8601 date-time`);
    issuedAt = undefined;
  }
  const style = text(value.style, `${path}.style`, errors);
  const aiGenerated = flag(value.aiGenerated, `${path}.aiGenerated`, errors);
  if (errors.length !== before) return;
  return { avatarId: avatarId!, ownerId: ownerId!, region: region!, issuedAt: issuedAt!, style: style!, aiGenerated: aiGenerated! };
}
