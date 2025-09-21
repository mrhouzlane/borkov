import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://jqwlfdmqehrlqpogirqo.supabase.co';
const supabaseAnonKey =
	'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impxd2xmZG1xZWhybHFwb2dpcnFvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc4NjMwMDUsImV4cCI6MjA3MzQzOTAwNX0.YPwg5QuLRaNFMiV6v-HCfCq1UskvX7ksaKXZ7ilrLik';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
