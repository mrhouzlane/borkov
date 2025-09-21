<script lang="ts">
	import { page } from '$app/state';
</script>

<main class="flex grow flex-col items-center justify-center">
	<h1 class="text-5xl font-bold">{page.status}</h1>
	<p class="text-xl">{page.error?.message}</p>
</main>
