import os

import click


baseDir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@click.group()
def main():
	pass


@main.command()
@click.option("-p","--project_dir_path", required=True, help="Path to project directory")
@click.option("-c","--config", required=True, help="Path to config file")
@click.option("-r","--run_as_admin", required=False, default=False, help="Should docker be run as root")
@click.option("-t","--tag", required=False, default='0.2.0', help="Should docker be run as root")
def run(project_dir_path,config,run_as_admin,tag):
	project_dir_path=os.path.abspath(project_dir_path)
	config=os.path.abspath(config)

	directive=''
	if run_as_admin:
		directive+='sudo '
	directive+=f'docker run --rm -v {baseDir}/runner:/home/runtime/runner '+\
		f'-v {baseDir}/user_interface:/home/runtime/user_interface -v {config}:/home/runtime/runner/user_config.json '+\
		f'-v {os.path.join(project_dir_path,"user_federates")}:/home/runtime/user_federates '+\
		f'-v {os.path.join(project_dir_path,"output")}:/home/output singlecontainerapp:{tag}'
	os.system(directive)


@main.command(name="create_project")
@click.option("-p","--project_dir_path", required=True, help="Path to template folder")
def create_project(project_dir_path):
	err=os.system(f'mkdir -p {os.path.join(project_dir_path,"config")} {os.path.join(project_dir_path,"output")}')
	assert err==0,f'creating project directory resulted in error:{err}'
	err=os.system(f'cp {os.path.join(baseDir,"runner","user_config.json")} {os.path.join(project_dir_path,"config")}')
	assert err==0,f'Copying config resulted in error:{err}'
	err=os.system(f'cp -r {os.path.join(baseDir,"user_federates")} {project_dir_path}')
	assert err==0,f'Copying config resulted in error:{err}'


@main.command(name="build")
@click.option("-t","--tag", required=True, help="Tag to be applied during docker build")
@click.option("-p","--python_cmd", required=False, default='python3' ,help="Python command to use i.e. python or python3")
def build(tag,python_cmd):
	err=os.system(f'{python_cmd} {os.path.join(baseDir,"build.py")} -t {tag}')
	assert err==0,f'Build resulted in error:{err} for directive={python_cmd} {os.path.join(baseDir,"build.py")} -t {tag}'


if __name__ == '__main__':
	main()
