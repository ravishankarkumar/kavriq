import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import katex from 'rehype-katex';
import remarkMath from 'remark-math';
import starlightSidebarTopics from 'starlight-sidebar-topics';

export default defineConfig({
	site: 'https://aiunderthehood.com',
	trailingSlash: 'never',
	build: {
		format: 'file'
	},
	integrations: [
		starlight({
			title: 'AI Under the Hood (Beta)',
			customCss: ['./src/styles/custom.css'],
			description: 'Learn Artificial Intelligence and Machine Learning with Rust through practical tutorials and projects.',
			logo: {
				src: './src/assets/new_logo_bg_removed.png',
			},
			lastUpdated: true,
			social: [
				{ label: 'LinkedIn', href: 'https://www.linkedin.com/in/ravi-shankar-a725b0225/', icon: 'linkedin' },
				{ label: 'GitHub', href: 'https://github.com/ravishankarkumar/aiunderthehood-sample-code', icon: 'github' }
			],
			head: [
				{ tag: 'link', attrs: { rel: 'icon', type: 'image/png', href: '/favicon.png' } },
				{ tag: 'meta', attrs: { name: 'keywords', content: 'Artificial Intelligence, Machine Learning, Rust AI, Rust ML, AI tutorial, deep learning, linear regression, Rust programming' } },
				{ tag: 'meta', attrs: { name: 'robots', content: 'index, follow' } },
				{ tag: 'meta', attrs: { name: 'og:title', content: 'AI Under the Hood: Learn AI/ML with Rust' } },
				{ tag: 'meta', attrs: { name: 'og:description', content: 'Explore AI and Machine Learning through Rust-based tutorials, projects, and visualizations.' } },
				{ tag: 'meta', attrs: { name: 'og:image', content: 'https://aiunderthehood.com/aiunderthehood.png' } },
				{ tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
			],
			plugins: [
				starlightSidebarTopics([
					// {
					// 	label: 'Home',
					// 	link: '/',
					// 	icon: 'seti:json',
					// 	items: []
					// },
					{
						label: 'ML Essentials',
						link: '/ml-essentials/introduction/overview',
						icon: 'open-book',
						items: [
							{
								label: 'Introduction',
								collapsed: false,
								autogenerate: { directory: 'ml-essentials/introduction' },
							},
							{
								label: 'Getting Started',
								collapsed: true,
								autogenerate: { directory: 'ml-essentials/getting-started' },
							},
							{
								label: 'Maths for AI/ML',
								collapsed: true,
								items: [
									{ label: 'Linear Algebra', autogenerate: { directory: 'ml-essentials/maths-for-aiml/linear-algebra' } },
									{ label: 'Calculus', autogenerate: { directory: 'ml-essentials/maths-for-aiml/calculus' } },
									{ label: 'Probability', autogenerate: { directory: 'ml-essentials/maths-for-aiml/probability' } },
									{ label: 'Statistics', autogenerate: { directory: 'ml-essentials/maths-for-aiml/statistics' } },
									{ label: 'Misc Math', autogenerate: { directory: 'ml-essentials/maths-for-aiml/misc-math' } },
								]
							},
							{ label: 'Core ML', collapsed: true, autogenerate: { directory: 'ml-essentials/core-ml' } },
							{ label: 'Deep Learning', collapsed: true, autogenerate: { directory: 'ml-essentials/deep-learning' } },
							{ label: 'Practical ML', collapsed: true, autogenerate: { directory: 'ml-essentials/practical-ml' } },
							{ label: 'Advanced Topics', collapsed: true, autogenerate: { directory: 'ml-essentials/advanced' } },
							{ label: 'Projects', collapsed: true, autogenerate: { directory: 'ml-essentials/projects' } },
							{ label: 'Resources', collapsed: true, autogenerate: { directory: 'ml-essentials/resources' } },
						],
					},
					{
						label: 'Agentic AI',
						link: '/agentic-ai',
						icon: 'rocket',
						items: [
							{ label: 'Core Concepts', autogenerate: { directory: 'agentic-ai' } },
						],
					},
					{
						label: 'Interview Prep',
						link: '/interview-prep',
						icon: 'comment-alt',
						items: [
							{ label: 'Prep Overview', autogenerate: { directory: 'interview-prep' } },
						],
					},
					{
						label: 'Blogs',
						link: '/blog',
						icon: 'pencil',
						items: [
							{ label: 'Latest Posts', autogenerate: { directory: 'blog' } },
						],
					},
					{
						label: 'Disclaimer',
						link: '/disclaimer',
						icon: 'warning',
						items: []
					},
				]),
			],
		}),
	],
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [katex],
	},
});