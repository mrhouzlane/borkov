import { Context } from 'runed';
import type { Session } from './session.svelte';
import type { components, operations } from '../generated/api';

export type Message = components['schemas']['ResolvedMessage'];

export type Thread = components['schemas']['ResolvedThread'];

export type AgentOption = {
	name: string;
	description?: string;
	value: string | undefined;
} & ({ type: 'string'; default: string | null } | { type: 'number'; default: number | null });

export type Agent = components['schemas']['SessionAgent'];
export type PublicRegistryAgent = components['schemas']['PublicRegistryAgent'];
export type Registry =
	operations['getAvailableAgents']['responses']['200']['content']['application/json'];

export const idAsKey = (id: PublicRegistryAgent['id']): string => `${id.name}${id.version}`;

export type GraphAgentRequest = components['schemas']['GraphAgentRequest'];

export type ToolTransport = {
	type: 'http';
	url: string;
};

export type CustomTool = components['schemas']['CustomTool'];

export const sessionCtx = new Context<{
	session: Session | null;
	registry: Registry | null;
	sessions: string[] | null;
	connection: { host: string; appId: string; privacyKey: string } | null;
}>('sessionCtx');
