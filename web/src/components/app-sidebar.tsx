import { Link } from "@tanstack/react-router";
import { LogOut } from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
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
      <SidebarSeparator />
      <SidebarFooter>
        <p className="truncate px-2 text-sm font-medium">{idLector}</p>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton render={<Link to="/" />}>
              <LogOut />
              <span>Salir</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
