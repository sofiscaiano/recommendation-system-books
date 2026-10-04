import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar";

type AppSidebarProps = {
  idLector: string;
};

export function AppSidebar({ idLector }: AppSidebarProps) {
  return (
    <Sidebar>
      <SidebarHeader>
        <div className="px-2 py-1.5 font-heading text-sm font-semibold">Qué Libro Leo</div>
      </SidebarHeader>
      <SidebarContent />
      <SidebarFooter>
        <NavUser name={idLector} />
      </SidebarFooter>
    </Sidebar>
  );
}
