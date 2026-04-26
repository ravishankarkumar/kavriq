export const SITE = {
  website: "https://kavriq.com/",
  author: "Ravi Shankar Kumar",
  profile: "https://www.linkedin.com/in/ravi-shankar-a725b0225/",
  desc: "KAVRIQ is a premium deep-dive learning platform for AI systems, agentic AI, machine learning, and production engineering.",
  title: "KAVRIQ",
  navTitle: "KAVRIQ",
  ogImage: "kavriq.jpg",
  googleAnalyticsId: "G-V8BNZV4KJF",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/ravishankarkumar/kavriq/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
