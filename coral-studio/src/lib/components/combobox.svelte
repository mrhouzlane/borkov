<script lang="ts" generics="Value">
	import CheckIcon from '@lucide/svelte/icons/check';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import { tick, type ComponentProps, type Snippet } from 'svelte';
	import * as Command from '$lib/components/ui/command/index.js';
	import * as Popover from '$lib/components/ui/popover/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { cn } from '$lib/utils.js';

	let {
		open = $bindable(false),
		selected = $bindable(undefined),
		selectPlaceholder = 'Select an item...',
		searchPlaceholder = 'Search items...',
		emptyLabel = 'No items found.',
		options = [],
		onValueChange,

		side,
		align,

		option: optionChild,
		trigger,
		class: className
	}: {
		open?: boolean;
		selected?: { label: string; key: string; value: Value } | undefined;
		options?: { label: string; key: string; value: Value }[];
		selectPlaceholder?: string;
		searchPlaceholder?: string;
		emptyLabel?: string;
		onValueChange?: (value: Value) => void;

		side?: ComponentProps<typeof Popover.Content>['side'];
		align?: ComponentProps<typeof Popover.Content>['align'];

		option?: Snippet<[{ option: { label: string; key: string; value: Value } }]>;
		trigger?: Snippet<[{ props: Record<string, unknown> }]>;
		class?: string;
	} = $props();

	let triggerRef = $state<HTMLButtonElement>(null!);
	// We want to refocus the trigger button when the user selects
	// an item from the list so users can continue navigating the
	// rest of the form with the keyboard.
	function closeAndFocusTrigger() {
		open = false;
		tick().then(() => {
			triggerRef.focus();
		});
	}
</script>

<Popover.Root bind:open>
	<Popover.Trigger bind:ref={triggerRef}>
		{#snippet child({ props })}
			{#if trigger}
				{@render trigger({ props })}
			{:else}
				<Button
					variant="outline"
					{...props}
					class={cn('w-[200px] justify-between', className)}
					role="combobox"
					aria-expanded={open}
				>
					{selected?.label || selectPlaceholder}
					<ChevronsUpDownIcon class="ml-2 size-4 shrink-0 opacity-50" />
				</Button>
			{/if}
		{/snippet}
	</Popover.Trigger>
	<Popover.Content class="w-[200px] p-0" {side} {align}>
		<Command.Root>
			<Command.Input placeholder={searchPlaceholder} />
			<Command.List>
				<Command.Empty>{emptyLabel}</Command.Empty>
				<Command.Group>
					{#each options as option}
						<Command.Item
							value={option.key}
							onSelect={() => {
								selected = option;
								onValueChange?.(option.value);
								closeAndFocusTrigger();
							}}
						>
							{#if optionChild}
								{@render optionChild({ option })}
							{:else}
								<CheckIcon
									class={cn('mr-2 size-4', selected?.key !== option.key && 'text-transparent')}
								/>
								{option.label}
							{/if}
						</Command.Item>
					{/each}
				</Command.Group>
			</Command.List>
		</Command.Root>
	</Popover.Content>
</Popover.Root>
