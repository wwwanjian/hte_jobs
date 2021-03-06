"""init database

Revision ID: fff0ce7ef26f
Revises: 
Create Date: 2018-05-16 00:03:49.184991

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fff0ce7ef26f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('mlpm_task_func',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('creator', sa.String(length=128), nullable=True),
    sa.Column('name', sa.String(length=128), nullable=True),
    sa.Column('desc', sa.VARCHAR(length=1023), nullable=True),
    sa.Column('doc', sa.VARCHAR(length=65535), nullable=True),
    sa.Column('pub_date', sa.TIMESTAMP(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_mlpm_task_func_id'), 'mlpm_task_func', ['id'], unique=False)
    op.create_table('user_task',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('username', sa.String(length=128), nullable=True),
    sa.Column('func_id', sa.BigInteger(), nullable=True),
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('args', sa.VARCHAR(length=1023), nullable=True),
    sa.Column('kwargs', sa.VARCHAR(length=1023), nullable=True),
    sa.Column('desc', sa.VARCHAR(length=1023), nullable=True),
    sa.Column('create_date', sa.TIMESTAMP(), nullable=True),
    sa.ForeignKeyConstraint(['func_id'], ['mlpm_task_func.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username', 'task_id')
    )
    op.create_index(op.f('ix_user_task_id'), 'user_task', ['id'], unique=False)
    op.create_index(op.f('ix_user_task_task_id'), 'user_task', ['task_id'], unique=False)
    op.create_index('ix_username_create_date', 'user_task', ['username', 'create_date'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_username_create_date', table_name='user_task')
    op.drop_index(op.f('ix_user_task_task_id'), table_name='user_task')
    op.drop_index(op.f('ix_user_task_id'), table_name='user_task')
    op.drop_table('user_task')
    op.drop_index(op.f('ix_mlpm_task_func_id'), table_name='mlpm_task_func')
    op.drop_table('mlpm_task_func')
    # ### end Alembic commands ###
