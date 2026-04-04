export const SITE = {
  website: "https://aiunderthehood.com/",
  author: "Ravi Shankar Kumar",
  profile: "https://www.linkedin.com/in/ravi-shankar-a725b0225/",
  desc: "Deep dives into AI internals — transformers, inference, embeddings, and everything under the hood.",
  title: "AI Under the Hood",
  navTitle: "AUTH",
  ogImage: "aiunderthehood.png",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/ravishankarkumar/aiunderthehood/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
