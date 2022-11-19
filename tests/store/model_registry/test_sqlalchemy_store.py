import os
import unittest

import tempfile
from unittest import mock
import uuid
import pytest

from mlflow.entities.model_registry import (
    ModelVersion,
    RegisteredModelTag,
    ModelVersionTag,
)
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.dbmodels.models import (
    SqlRegisteredModel,
    SqlRegisteredModelTag,
    SqlModelVersion,
    SqlModelVersionTag,
)
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
)
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from tests.helper_functions import random_str

DB_URI = "sqlite:///"

pytestmark = pytest.mark.notrackingurimock


class TestSqlAlchemyStoreSqlite(unittest.TestCase):
    def _get_store(self, db_uri=""):
        return SqlAlchemyStore(db_uri)

    def _setup_db_uri(self):
        if _TRACKING_URI_ENV_VAR in os.environ:
            self.temp_dbfile = None
            self.db_url = os.getenv(_TRACKING_URI_ENV_VAR)
        else:
            fd, self.temp_dbfile = tempfile.mkstemp()
            # Close handle immediately so that we can remove the file later on in Windows
            os.close(fd)
            self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)

    def setUp(self):
        self._setup_db_uri()
        self.store = self._get_store(self.db_url)

    def get_store(self):
        return self.store

    def tearDown(self):
        if self.temp_dbfile:
            os.remove(self.temp_dbfile)
        else:
            with self.store.ManagedSessionMaker() as session:
                for model in (
                    SqlModelVersionTag,
                    SqlRegisteredModelTag,
                    SqlModelVersion,
                    SqlRegisteredModel,
                ):
                    session.query(model).delete()

    def _rm_maker(self, name, tags=None, description=None):
        return self.store.create_registered_model(name, tags, description)

    def _mv_maker(
        self,
        name,
        source="path/to/source",
        run_id=uuid.uuid4().hex,
        tags=None,
        run_link=None,
        description=None,
    ):
        return self.store.create_model_version(
            name, source, run_id, tags, run_link=run_link, description=description
        )

    def _extract_latest_by_stage(self, latest_versions):
        return {mvd.current_stage: mvd.version for mvd in latest_versions}

    def test_create_registered_model(self):
        name = random_str() + "abCD"
        rm1 = self._rm_maker(name)
        assert rm1.name == name
        assert rm1.description is None

        # error on duplicate
        with pytest.raises(
            MlflowException, match=rf"Registered Model \(name={name}\) already exists"
        ) as exception_context:
            self._rm_maker(name)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

        # slightly different name is ok
        for name2 in [name + "extra", name + name]:
            rm2 = self._rm_maker(name2)
            assert rm2.name == name2

        # test create model with tags
        name2 = random_str() + "tags"
        tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        rm2 = self._rm_maker(name2, tags)
        rmd2 = self.store.get_registered_model(name2)
        assert rm2.name == name2
        assert rm2.tags == {tag.key: tag.value for tag in tags}
        assert rmd2.name == name2
        assert rmd2.tags == {tag.key: tag.value for tag in tags}

        # create with description
        name3 = random_str() + "-description"
        description = "the best model ever"
        rm3 = self._rm_maker(name3, description=description)
        rmd3 = self.store.get_registered_model(name3)
        assert rm3.name == name3
        assert rm3.description == description
        assert rmd3.name == name3
        assert rmd3.description == description

        # invalid model name will fail
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self._rm_maker(None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self._rm_maker("")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_get_registered_model(self):
        name = "model_1"
        tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        # use fake clock
        with mock.patch("time.time") as mock_time:
            mock_time.return_value = 1234
            rm = self._rm_maker(name, tags)
            assert rm.name == name
        rmd = self.store.get_registered_model(name=name)
        assert rmd.name == name
        assert rmd.creation_timestamp == 1234000
        assert rmd.last_updated_timestamp == 1234000
        assert rmd.description is None
        assert rmd.latest_versions == []
        assert rmd.tags == {tag.key: tag.value for tag in tags}

    def test_update_registered_model(self):
        name = "model_for_update_RM"
        rm1 = self._rm_maker(name)
        rmd1 = self.store.get_registered_model(name=name)
        assert rm1.name == name
        assert rmd1.description is None

        # update description
        rm2 = self.store.update_registered_model(name=name, description="test model")
        rmd2 = self.store.get_registered_model(name=name)
        assert rm2.name == "model_for_update_RM"
        assert rmd2.name == "model_for_update_RM"
        assert rmd2.description == "test model"

    def test_rename_registered_model(self):
        original_name = "original name"
        new_name = "new name"
        self._rm_maker(original_name)
        self._mv_maker(original_name)
        self._mv_maker(original_name)
        rm = self.store.get_registered_model(original_name)
        mv1 = self.store.get_model_version(original_name, 1)
        mv2 = self.store.get_model_version(original_name, 2)
        assert rm.name == original_name
        assert mv1.name == original_name
        assert mv2.name == original_name

        # test renaming registered model also updates its model versions
        self.store.rename_registered_model(original_name, new_name)
        rm = self.store.get_registered_model(new_name)
        mv1 = self.store.get_model_version(new_name, 1)
        mv2 = self.store.get_model_version(new_name, 2)
        assert rm.name == new_name
        assert mv1.name == new_name
        assert mv2.name == new_name

        # test accessing the model with the old name will fail
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={original_name} not found"
        ) as exception_context:
            self.store.get_registered_model(original_name)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # test name another model with the replaced name is ok
        self._rm_maker(original_name)
        # cannot rename model to conflict with an existing model
        with pytest.raises(
            MlflowException, match=rf"Registered Model \(name={original_name}\) already exists"
        ) as exception_context:
            self.store.rename_registered_model(new_name, original_name)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)
        # invalid model name will fail
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.rename_registered_model(original_name, None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.rename_registered_model(original_name, "")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_delete_registered_model(self):
        name = "model_for_delete_RM"
        self._rm_maker(name)
        self._mv_maker(name)
        rm1 = self.store.get_registered_model(name=name)
        mv1 = self.store.get_model_version(name, 1)
        assert rm1.name == name
        assert mv1.name == name

        # delete model
        self.store.delete_registered_model(name=name)

        # cannot get model
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={name} not found"
        ) as exception_context:
            self.store.get_registered_model(name=name)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot update a delete model
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={name} not found"
        ) as exception_context:
            self.store.update_registered_model(name=name, description="deleted")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot delete it again
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={name} not found"
        ) as exception_context:
            self.store.delete_registered_model(name=name)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # model versions are cascade deleted with the registered model
        with pytest.raises(
            MlflowException, match=rf"Model Version \(name={name}, version=1\) not found"
        ) as exception_context:
            self.store.get_model_version(name, 1)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_get_latest_versions(self):
        name = "test_for_latest_versions"
        self._rm_maker(name)
        rmd1 = self.store.get_registered_model(name=name)
        assert rmd1.latest_versions == []

        mv1 = self._mv_maker(name)
        assert mv1.version == 1
        rmd2 = self.store.get_registered_model(name=name)
        assert self._extract_latest_by_stage(rmd2.latest_versions) == {"None": 1}

        # add a bunch more
        mv2 = self._mv_maker(name)
        assert mv2.version == 2
        self.store.transition_model_version_stage(
            name=mv2.name, version=mv2.version, stage="Production", archive_existing_versions=False
        )

        mv3 = self._mv_maker(name)
        assert mv3.version == 3
        self.store.transition_model_version_stage(
            name=mv3.name, version=mv3.version, stage="Production", archive_existing_versions=False
        )
        mv4 = self._mv_maker(name)
        assert mv4.version == 4
        self.store.transition_model_version_stage(
            name=mv4.name, version=mv4.version, stage="Staging", archive_existing_versions=False
        )

        # test that correct latest versions are returned for each stage
        rmd4 = self.store.get_registered_model(name=name)
        assert self._extract_latest_by_stage(rmd4.latest_versions) == {
            "None": 1,
            "Production": 3,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=None)
        ) == {"None": 1, "Production": 3, "Staging": 4}
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=[])
        ) == {"None": 1, "Production": 3, "Staging": 4}
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=["Production"])
        ) == {"Production": 3}
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=["production"])
        ) == {
            "Production": 3
        }  # The stages are case insensitive.
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=["pROduction"])
        ) == {
            "Production": 3
        }  # The stages are case insensitive.
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=["None", "Production"])
        ) == {"None": 1, "Production": 3}

        # delete latest Production, and should point to previous one
        self.store.delete_model_version(name=mv3.name, version=mv3.version)
        rmd5 = self.store.get_registered_model(name=name)
        assert self._extract_latest_by_stage(rmd5.latest_versions) == {
            "None": 1,
            "Production": 2,
            "Staging": 4,
        }
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=None)
        ) == {"None": 1, "Production": 2, "Staging": 4}
        assert self._extract_latest_by_stage(
            self.store.get_latest_versions(name=name, stages=["Production"])
        ) == {"Production": 2}

    def test_set_registered_model_tag(self):
        name1 = "SetRegisteredModelTag_TestMod"
        name2 = "SetRegisteredModelTag_TestMod 2"
        initial_tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        self._rm_maker(name1, initial_tags)
        self._rm_maker(name2, initial_tags)
        new_tag = RegisteredModelTag("randomTag", "not a random value")
        self.store.set_registered_model_tag(name1, new_tag)
        rm1 = self.store.get_registered_model(name=name1)
        all_tags = initial_tags + [new_tag]
        assert rm1.tags == {tag.key: tag.value for tag in all_tags}

        # test overriding a tag with the same key
        overriding_tag = RegisteredModelTag("key", "overriding")
        self.store.set_registered_model_tag(name1, overriding_tag)
        all_tags = [tag for tag in all_tags if tag.key != "key"] + [overriding_tag]
        rm1 = self.store.get_registered_model(name=name1)
        assert rm1.tags == {tag.key: tag.value for tag in all_tags}
        # does not affect other models with the same key
        rm2 = self.store.get_registered_model(name=name2)
        assert rm2.tags == {tag.key: tag.value for tag in initial_tags}

        # can not set tag on deleted (non-existed) registered model
        self.store.delete_registered_model(name1)
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={name1} not found"
        ) as exception_context:
            self.store.set_registered_model_tag(name1, overriding_tag)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # test cannot set tags that are too long
        long_tag = RegisteredModelTag("longTagKey", "a" * 5001)
        with pytest.raises(
            MlflowException,
            match=(
                r"Registered model value '.+' had length \d+, which exceeded length limit of 5000"
            ),
        ) as exception_context:
            self.store.set_registered_model_tag(name2, long_tag)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test can set tags that are somewhat long
        long_tag = RegisteredModelTag("longTagKey", "a" * 4999)
        self.store.set_registered_model_tag(name2, long_tag)
        # can not set invalid tag
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            self.store.set_registered_model_tag(name2, RegisteredModelTag(key=None, value=""))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.set_registered_model_tag(None, RegisteredModelTag(key="key", value="value"))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_delete_registered_model_tag(self):
        name1 = "DeleteRegisteredModelTag_TestMod"
        name2 = "DeleteRegisteredModelTag_TestMod 2"
        initial_tags = [
            RegisteredModelTag("key", "value"),
            RegisteredModelTag("anotherKey", "some other value"),
        ]
        self._rm_maker(name1, initial_tags)
        self._rm_maker(name2, initial_tags)
        new_tag = RegisteredModelTag("randomTag", "not a random value")
        self.store.set_registered_model_tag(name1, new_tag)
        self.store.delete_registered_model_tag(name1, "randomTag")
        rm1 = self.store.get_registered_model(name=name1)
        assert rm1.tags == {tag.key: tag.value for tag in initial_tags}

        # testing deleting a key does not affect other models with the same key
        self.store.delete_registered_model_tag(name1, "key")
        rm1 = self.store.get_registered_model(name=name1)
        rm2 = self.store.get_registered_model(name=name2)
        assert rm1.tags == {"anotherKey": "some other value"}
        assert rm2.tags == {tag.key: tag.value for tag in initial_tags}

        # delete tag that is already deleted does nothing
        self.store.delete_registered_model_tag(name1, "key")
        rm1 = self.store.get_registered_model(name=name1)
        assert rm1.tags == {"anotherKey": "some other value"}

        # can not delete tag on deleted (non-existed) registered model
        self.store.delete_registered_model(name1)
        with pytest.raises(
            MlflowException, match=rf"Registered Model with name={name1} not found"
        ) as exception_context:
            self.store.delete_registered_model_tag(name1, "anotherKey")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # can not delete tag with invalid key
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            self.store.delete_registered_model_tag(name2, None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.delete_registered_model_tag(None, "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_create_model_version(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        run_id = uuid.uuid4().hex
        with mock.patch("time.time") as mock_time:
            mock_time.return_value = 456778
            mv1 = self._mv_maker(name, "a/b/CD", run_id)
            assert mv1.name == name
            assert mv1.version == 1

        mvd1 = self.store.get_model_version(mv1.name, mv1.version)
        assert mvd1.name == name
        assert mvd1.version == 1
        assert mvd1.current_stage == "None"
        assert mvd1.creation_timestamp == 456778000
        assert mvd1.last_updated_timestamp == 456778000
        assert mvd1.description is None
        assert mvd1.source == "a/b/CD"
        assert mvd1.run_id == run_id
        assert mvd1.status == "READY"
        assert mvd1.status_message is None
        assert mvd1.tags == {}

        # new model versions for same name autoincrement versions
        mv2 = self._mv_maker(name)
        mvd2 = self.store.get_model_version(name=mv2.name, version=mv2.version)
        assert mv2.version == 2
        assert mvd2.version == 2

        # create model version with tags return model version entity with tags
        tags = [ModelVersionTag("key", "value"), ModelVersionTag("anotherKey", "some other value")]
        mv3 = self._mv_maker(name, tags=tags)
        mvd3 = self.store.get_model_version(name=mv3.name, version=mv3.version)
        assert mv3.version == 3
        assert mv3.tags == {tag.key: tag.value for tag in tags}
        assert mvd3.version == 3
        assert mvd3.tags == {tag.key: tag.value for tag in tags}

        # create model versions with runLink
        run_link = "http://localhost:3000/path/to/run/"
        mv4 = self._mv_maker(name, run_link=run_link)
        mvd4 = self.store.get_model_version(name, mv4.version)
        assert mv4.version == 4
        assert mv4.run_link == run_link
        assert mvd4.version == 4
        assert mvd4.run_link == run_link

        # create model version with description
        description = "the best model ever"
        mv5 = self._mv_maker(name, description=description)
        mvd5 = self.store.get_model_version(name, mv5.version)
        assert mv5.version == 5
        assert mv5.description == description
        assert mvd5.version == 5
        assert mvd5.description == description

        # create model version without runId
        mv6 = self._mv_maker(name, run_id=None)
        mvd6 = self.store.get_model_version(name, mv6.version)
        assert mv6.version == 6
        assert mv6.run_id is None
        assert mvd6.version == 6
        assert mvd6.run_id is None

    def test_update_model_version(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        mv1 = self._mv_maker(name)
        mvd1 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd1.name == name
        assert mvd1.version == 1
        assert mvd1.current_stage == "None"

        # update stage
        self.store.transition_model_version_stage(
            name=mv1.name, version=mv1.version, stage="Production", archive_existing_versions=False
        )
        mvd2 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd2.name == name
        assert mvd2.version == 1
        assert mvd2.current_stage == "Production"
        assert mvd2.description is None

        # update description
        self.store.update_model_version(
            name=mv1.name, version=mv1.version, description="test model version"
        )
        mvd3 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        assert mvd3.name == name
        assert mvd3.version == 1
        assert mvd3.current_stage == "Production"
        assert mvd3.description == "test model version"

        # only valid stages can be set
        with pytest.raises(
            MlflowException,
            match=(
                "Invalid Model Version stage: unknown. "
                "Value must be one of None, Staging, Production, Archived."
            ),
        ) as exception_context:
            self.store.transition_model_version_stage(
                mv1.name, mv1.version, stage="unknown", archive_existing_versions=False
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # stages are case-insensitive and auto-corrected to system stage names
        for stage_name in ["STAGING", "staging", "StAgInG"]:
            self.store.transition_model_version_stage(
                name=mv1.name,
                version=mv1.version,
                stage=stage_name,
                archive_existing_versions=False,
            )
            mvd5 = self.store.get_model_version(name=mv1.name, version=mv1.version)
            assert mvd5.current_stage == "Staging"

    def test_transition_model_version_stage_when_archive_existing_versions_is_false(self):
        name = "model"
        self._rm_maker(name)
        mv1 = self._mv_maker(name)
        mv2 = self._mv_maker(name)
        mv3 = self._mv_maker(name)

        # test that when `archive_existing_versions` is False, transitioning a model version
        # to the inactive stages ("Archived" and "None") does not throw.
        for stage in ["Archived", "None"]:
            self.store.transition_model_version_stage(name, mv1.version, stage, False)

        self.store.transition_model_version_stage(name, mv1.version, "Staging", False)
        self.store.transition_model_version_stage(name, mv2.version, "Production", False)
        self.store.transition_model_version_stage(name, mv3.version, "Staging", False)

        mvd1 = self.store.get_model_version(name=name, version=mv1.version)
        mvd2 = self.store.get_model_version(name=name, version=mv2.version)
        mvd3 = self.store.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Staging"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Staging"

        self.store.transition_model_version_stage(name, mv3.version, "Production", False)

        mvd1 = self.store.get_model_version(name=name, version=mv1.version)
        mvd2 = self.store.get_model_version(name=name, version=mv2.version)
        mvd3 = self.store.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Staging"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Production"

    def test_transition_model_version_stage_when_archive_existing_versions_is_true(self):
        name = "model"
        self._rm_maker(name)
        mv1 = self._mv_maker(name)
        mv2 = self._mv_maker(name)
        mv3 = self._mv_maker(name)

        msg = (
            r"Model version transition cannot archive existing model versions "
            r"because .+ is not an Active stage"
        )

        # test that when `archive_existing_versions` is True, transitioning a model version
        # to the inactive stages ("Archived" and "None") throws.
        for stage in ["Archived", "None"]:
            with pytest.raises(MlflowException, match=msg):
                self.store.transition_model_version_stage(name, mv1.version, stage, True)

        self.store.transition_model_version_stage(name, mv1.version, "Staging", False)
        self.store.transition_model_version_stage(name, mv2.version, "Production", False)
        self.store.transition_model_version_stage(name, mv3.version, "Staging", True)

        mvd1 = self.store.get_model_version(name=name, version=mv1.version)
        mvd2 = self.store.get_model_version(name=name, version=mv2.version)
        mvd3 = self.store.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Archived"
        assert mvd2.current_stage == "Production"
        assert mvd3.current_stage == "Staging"
        assert mvd1.last_updated_timestamp == mvd3.last_updated_timestamp

        self.store.transition_model_version_stage(name, mv3.version, "Production", True)

        mvd1 = self.store.get_model_version(name=name, version=mv1.version)
        mvd2 = self.store.get_model_version(name=name, version=mv2.version)
        mvd3 = self.store.get_model_version(name=name, version=mv3.version)

        assert mvd1.current_stage == "Archived"
        assert mvd2.current_stage == "Archived"
        assert mvd3.current_stage == "Production"
        assert mvd2.last_updated_timestamp == mvd3.last_updated_timestamp

        for uncanonical_stage_name in ["STAGING", "staging", "StAgInG"]:
            self.store.transition_model_version_stage(mv1.name, mv1.version, "Staging", False)
            self.store.transition_model_version_stage(mv2.name, mv2.version, "None", False)

            # stage names are case-insensitive and auto-corrected to system stage names
            self.store.transition_model_version_stage(
                mv2.name, mv2.version, uncanonical_stage_name, True
            )

            mvd1 = self.store.get_model_version(name=mv1.name, version=mv1.version)
            mvd2 = self.store.get_model_version(name=mv2.name, version=mv2.version)
            assert mvd1.current_stage == "Archived"
            assert mvd2.current_stage == "Staging"

    def test_delete_model_version(self):
        name = "test_for_delete_MV"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        self._rm_maker(name)
        mv = self._mv_maker(name, tags=initial_tags)
        mvd = self.store.get_model_version(name=mv.name, version=mv.version)
        assert mvd.name == name

        self.store.delete_model_version(name=mv.name, version=mv.version)

        # cannot get a deleted model version
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            self.store.get_model_version(name=mv.name, version=mv.version)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot update a delete
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            self.store.update_model_version(mv.name, mv.version, description="deleted!")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot delete it again
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            self.store.delete_model_version(name=mv.name, version=mv.version)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_delete_model_version_redaction(self):
        name = "test_for_delete_MV_redaction"
        run_link = "http://localhost:5000/path/to/run"
        run_id = "12345"
        source = "path/to/source"
        self._rm_maker(name)
        mv = self._mv_maker(name, source=source, run_id=run_id, run_link=run_link)
        mvd = self.store.get_model_version(name=name, version=mv.version)
        assert mvd.run_link == run_link
        assert mvd.run_id == run_id
        assert mvd.source == source
        # delete the MV now
        self.store.delete_model_version(name, mv.version)
        # verify that the relevant fields are redacted
        mvd_deleted = self.store._get_sql_model_version_including_deleted(
            name=name, version=mv.version
        )
        assert "REDACTED" in mvd_deleted.run_link
        assert "REDACTED" in mvd_deleted.source
        assert "REDACTED" in mvd_deleted.run_id

    def test_get_model_version_download_uri(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        source_path = "path/to/source"
        mv = self._mv_maker(name, source=source_path, run_id=uuid.uuid4().hex)
        mvd1 = self.store.get_model_version(name=mv.name, version=mv.version)
        assert mvd1.name == name
        assert mvd1.source == source_path

        # download location points to source
        assert (
            self.store.get_model_version_download_uri(name=mv.name, version=mv.version)
            == source_path
        )

        # download URI does not change even if model version is updated
        self.store.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="Production", archive_existing_versions=False
        )
        self.store.update_model_version(
            name=mv.name, version=mv.version, description="Test for Path"
        )
        mvd2 = self.store.get_model_version(name=mv.name, version=mv.version)
        assert mvd2.source == source_path
        assert (
            self.store.get_model_version_download_uri(name=mv.name, version=mv.version)
            == source_path
        )

        # cannot retrieve download URI for deleted model versions
        self.store.delete_model_version(name=mv.name, version=mv.version)
        with pytest.raises(
            MlflowException,
            match=rf"Model Version \(name={mv.name}, version={mv.version}\) not found",
        ) as exception_context:
            self.store.get_model_version_download_uri(name=mv.name, version=mv.version)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_search_model_versions(self):
        # create some model versions
        name = "test_for_search_MV"
        self._rm_maker(name)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        mv1 = self._mv_maker(name=name, source="A/B", run_id=run_id_1)
        assert mv1.version == 1
        mv2 = self._mv_maker(name=name, source="A/C", run_id=run_id_2)
        assert mv2.version == 2
        mv3 = self._mv_maker(name=name, source="A/D", run_id=run_id_2)
        assert mv3.version == 3
        mv4 = self._mv_maker(name=name, source="A/D", run_id=run_id_3)
        assert mv4.version == 4

        def search_versions(filter_string):
            return [mvd.version for mvd in self.store.search_model_versions(filter_string)]

        # search using name should return all 4 versions
        assert set(search_versions("name='%s'" % name)) == {1, 2, 3, 4}

        # search using run_id_1 should return version 1
        assert set(search_versions("run_id='%s'" % run_id_1)) == {1}

        # search using run_id_2 should return versions 2 and 3
        assert set(search_versions("run_id='%s'" % run_id_2)) == {2, 3}

        # search using the IN operator should return all versions
        assert set(search_versions(f"run_id IN ('{run_id_1}','{run_id_2}')")) == {1, 2, 3}

        # search IN operator is case sensitive
        assert set(search_versions(f"run_id IN ('{run_id_1.upper()}','{run_id_2}')")) == {2, 3}

        # search IN operator with right-hand side value containing whitespaces
        assert set(search_versions(f"run_id IN ('{run_id_1}', '{run_id_2}')")) == {1, 2, 3}

        # search using the IN operator with bad lists should return exceptions
        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected string value, punctuation, or whitespace, "
                r"but got different type in list"
            ),
        ) as exception_context:
            search_versions("run_id IN (1,2,3)")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        assert set(search_versions(f"run_id LIKE '{run_id_2[:30]}%'")) == {2, 3}

        assert set(search_versions(f"run_id ILIKE '{run_id_2[:30].upper()}%'")) == {2, 3}

        # search using the IN operator with empty lists should return exceptions
        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got empty list"
            ),
        ) as exception_context:
            search_versions("run_id IN ()")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # search using an ill-formed IN operator correctly throws exception
        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("run_id IN (")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("run_id IN")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            search_versions("name LIKE")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got ill-formed list"
            ),
        ) as exception_context:
            search_versions("run_id IN (,)")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        with pytest.raises(
            MlflowException,
            match=(
                r"While parsing a list in the query, "
                r"expected a non-empty list of string values, "
                r"but got ill-formed list"
            ),
        ) as exception_context:
            search_versions("run_id IN ('runid1',,'runid2')")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # search using source_path "A/D" should return version 3 and 4
        assert set(search_versions("source_path = 'A/D'")) == {3, 4}

        # search using source_path "A" should not return anything
        assert len(search_versions("source_path = 'A'")) == 0
        assert len(search_versions("source_path = 'A/'")) == 0
        assert len(search_versions("source_path = ''")) == 0

        # delete mv4. search should not return version 4
        self.store.delete_model_version(name=mv4.name, version=mv4.version)
        assert set(search_versions("")) == {1, 2, 3}

        assert set(search_versions(None)) == {1, 2, 3}

        assert set(search_versions("name='%s'" % name)) == {1, 2, 3}

        assert set(search_versions("source_path = 'A/D'")) == {3}

        self.store.transition_model_version_stage(
            name=mv1.name, version=mv1.version, stage="production", archive_existing_versions=False
        )

        self.store.update_model_version(
            name=mv1.name, version=mv1.version, description="Online prediction model!"
        )

        mvds = self.store.search_model_versions("run_id = '%s'" % run_id_1)
        assert len(mvds) == 1
        assert isinstance(mvds[0], ModelVersion)
        assert mvds[0].current_stage == "Production"
        assert mvds[0].run_id == run_id_1
        assert mvds[0].source == "A/B"
        assert mvds[0].description == "Online prediction model!"

    def test_search_model_versions_by_tag(self):
        # create some model versions
        name = "test_for_search_MV_by_tag"
        self._rm_maker(name)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex

        mv1 = self._mv_maker(
            name=name,
            source="A/B",
            run_id=run_id_1,
            tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "xyz")],
        )
        assert mv1.version == 1
        mv2 = self._mv_maker(
            name=name,
            source="A/C",
            run_id=run_id_2,
            tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "x123")],
        )
        assert mv2.version == 2

        def search_versions(filter_string):
            return [mvd.version for mvd in self.store.search_model_versions(filter_string)]

        assert search_versions(f"name = '{name}' and tag.t2 = 'xyz'") == [1]
        assert search_versions("name = 'wrong_name' and tag.t2 = 'xyz'") == []
        assert search_versions("tag.`t2` = 'xyz'") == [1]
        assert search_versions("tag.t3 = 'xyz'") == []
        assert search_versions("tag.t2 != 'xy'") == [2, 1]
        assert search_versions("tag.t2 LIKE 'xy%'") == [1]
        assert search_versions("tag.t2 LIKE 'xY%'") == []
        assert search_versions("tag.t2 ILIKE 'xY%'") == [1]
        assert search_versions("tag.t2 LIKE 'x%'") == [2, 1]
        assert search_versions("tag.T2 = 'xyz'") == []
        assert search_versions("tag.t1 = 'abc' and tag.t2 = 'xyz'") == [1]
        assert search_versions("tag.t1 = 'abc' and tag.t2 LIKE 'x%'") == [2, 1]
        assert search_versions("tag.t1 = 'abc' and tag.t2 LIKE 'y%'") == []
        # test filter with duplicated keys
        assert search_versions("tag.t2 like 'x%' and tag.t2 != 'xyz'") == [2]

    def _search_registered_models(
        self, filter_string, max_results=10, order_by=None, page_token=None
    ):
        result = self.store.search_registered_models(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )
        return [registered_model.name for registered_model in result], result.token

    def test_search_registered_models(self):
        # create some registered models
        prefix = "test_for_search_"
        names = [prefix + name for name in ["RM1", "RM2", "RM3", "RM4", "RM4A", "RM4ab"]]
        for name in names:
            self._rm_maker(name)

        # search with no filter should return all registered models
        rms, _ = self._search_registered_models(None)
        assert rms == names

        # equality search using name should return exactly the 1 name
        rms, _ = self._search_registered_models("name='{}'".format(names[0]))
        assert rms == [names[0]]

        # equality search using name that is not valid should return nothing
        rms, _ = self._search_registered_models("name='{}'".format(names[0] + "cats"))
        assert rms == []

        # case-sensitive prefix search using LIKE should return all the RMs
        rms, _ = self._search_registered_models("name LIKE '{}%'".format(prefix))
        assert rms == names

        # case-sensitive prefix search using LIKE with surrounding % should return all the RMs
        rms, _ = self._search_registered_models("name LIKE '%RM%'")
        assert rms == names

        # case-sensitive prefix search using LIKE with surrounding % should return all the RMs
        # _e% matches test_for_search_ , so all RMs should match
        rms, _ = self._search_registered_models("name LIKE '_e%'")
        assert rms == names

        # case-sensitive prefix search using LIKE should return just rm4
        rms, _ = self._search_registered_models("name LIKE '{}%'".format(prefix + "RM4A"))
        assert rms == [names[4]]

        # case-sensitive prefix search using LIKE should return no models if no match
        rms, _ = self._search_registered_models("name LIKE '{}%'".format(prefix + "cats"))
        assert rms == []

        # confirm that LIKE is not case-sensitive
        rms, _ = self._search_registered_models("name lIkE '%blah%'")
        assert rms == []

        rms, _ = self._search_registered_models("name like '{}%'".format(prefix + "RM4A"))
        assert rms == [names[4]]

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        rms, _ = self._search_registered_models("name ILIKE '{}%'".format(prefix + "RM4A"))
        assert rms == names[4:]

        # case-insensitive postfix search with ILIKE
        rms, _ = self._search_registered_models("name ILIKE '%RM4a%'")
        assert rms == names[4:]

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        rms, _ = self._search_registered_models("name ILIKE '{}%'".format(prefix + "cats"))
        assert rms == []

        # confirm that ILIKE is not case-sensitive
        rms, _ = self._search_registered_models("name iLike '%blah%'")
        assert rms == []

        # confirm that ILIKE works for empty query
        rms, _ = self._search_registered_models("name iLike '%%'")
        assert rms == names

        rms, _ = self._search_registered_models("name ilike '%RM4a%'")
        assert rms == names[4:]

        # cannot search by invalid comparator types
        with pytest.raises(
            MlflowException,
            match="Parameter value is either not quoted or unidentified quote types used for "
            "string value something",
        ) as exception_context:
            self._search_registered_models("name!=something")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by run_id
        with pytest.raises(
            MlflowException, match=r"Invalid attribute key 'run_id' specified."
        ) as exception_context:
            self._search_registered_models("run_id='%s'" % "somerunID")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by source_path
        with pytest.raises(
            MlflowException, match=r"Invalid attribute key 'source_path' specified."
        ) as exception_context:
            self._search_registered_models("source_path = 'A/D'")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # cannot search by other params
        with pytest.raises(
            MlflowException, match=r"Invalid clause\(s\) in filter string"
        ) as exception_context:
            self._search_registered_models("evilhax = true")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # delete last registered model. search should not return the first 5
        self.store.delete_registered_model(name=names[-1])
        assert self._search_registered_models(None, max_results=1000) == (names[:-1], None)

        # equality search using name should return no names
        assert self._search_registered_models("name='{}'".format(names[-1])) == ([], None)

        # case-sensitive prefix search using LIKE should return all the RMs
        assert self._search_registered_models("name LIKE '{}%'".format(prefix)) == (
            names[0:5],
            None,
        )

        # case-insensitive prefix search using ILIKE should return both rm5 and rm6
        assert self._search_registered_models("name ILIKE '{}%'".format(prefix + "RM4A")) == (
            [names[4]],
            None,
        )

    def test_search_registered_models_by_tag(self):
        name1 = "test_for_search_RM_by_tag1"
        name2 = "test_for_search_RM_by_tag2"
        tags1 = [
            RegisteredModelTag("t1", "abc"),
            RegisteredModelTag("t2", "xyz"),
        ]
        tags2 = [
            RegisteredModelTag("t1", "abcd"),
            RegisteredModelTag("t2", "xyz123"),
            RegisteredModelTag("t3", "XYZ"),
        ]
        self._rm_maker(name1, tags1)
        self._rm_maker(name2, tags2)

        rms, _ = self._search_registered_models("tag.t3 = 'XYZ'")
        assert rms == [name2]

        rms, _ = self._search_registered_models(f"name = '{name1}' and tag.t1 = 'abc'")
        assert rms == [name1]

        rms, _ = self._search_registered_models("tag.t1 LIKE 'ab%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models("tag.t1 ILIKE 'aB%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models("tag.t1 LIKE 'ab%' AND tag.t2 LIKE 'xy%'")
        assert rms == [name1, name2]

        rms, _ = self._search_registered_models("tag.t3 = 'XYz'")
        assert rms == []

        rms, _ = self._search_registered_models("tag.T3 = 'XYZ'")
        assert rms == []

        rms, _ = self._search_registered_models("tag.t1 != 'abc'")
        assert rms == [name2]

        # test filter with duplicated keys
        rms, _ = self._search_registered_models("tag.t1 != 'abcd' and tag.t1 LIKE 'ab%'")
        assert rms == [name1]

    def test_parse_search_registered_models_order_by(self):
        # test that "registered_models.name ASC" is returned by default
        parsed = SqlAlchemyStore._parse_search_registered_models_order_by([])
        assert [str(x) for x in parsed] == ["registered_models.name ASC"]

        # test that the given 'name' replaces the default one ('registered_models.name ASC')
        parsed = SqlAlchemyStore._parse_search_registered_models_order_by(["name DESC"])
        assert [str(x) for x in parsed] == ["registered_models.name DESC"]

        # test that an exception is raised when order_by contains duplicate fields
        msg = "`order_by` contains duplicate fields:"
        with pytest.raises(MlflowException, match=msg):
            SqlAlchemyStore._parse_search_registered_models_order_by(
                ["last_updated_timestamp", "last_updated_timestamp"]
            )

        with pytest.raises(MlflowException, match=msg):
            SqlAlchemyStore._parse_search_registered_models_order_by(["timestamp", "timestamp"])

        with pytest.raises(MlflowException, match=msg):
            SqlAlchemyStore._parse_search_registered_models_order_by(
                ["timestamp", "last_updated_timestamp"],
            )

        with pytest.raises(MlflowException, match=msg):
            SqlAlchemyStore._parse_search_registered_models_order_by(
                ["last_updated_timestamp ASC", "last_updated_timestamp DESC"],
            )

        with pytest.raises(MlflowException, match=msg):
            SqlAlchemyStore._parse_search_registered_models_order_by(
                ["last_updated_timestamp", "last_updated_timestamp DESC"],
            )

    def test_search_registered_model_pagination(self):
        rms = [self._rm_maker("RM{:03}".format(i)).name for i in range(50)]

        # test flow with fixed max_results
        returned_rms = []
        query = "name LIKE 'RM%'"
        result, token = self._search_registered_models(query, page_token=None, max_results=5)
        returned_rms.extend(result)
        while token:
            result, token = self._search_registered_models(query, page_token=token, max_results=5)
            returned_rms.extend(result)
        assert rms == returned_rms

        # test that pagination will return all valid results in sorted order
        # by name ascending
        result, token1 = self._search_registered_models(query, max_results=5)
        assert token1 is not None
        assert result == rms[0:5]

        result, token2 = self._search_registered_models(query, page_token=token1, max_results=10)
        assert token2 is not None
        assert result == rms[5:15]

        result, token3 = self._search_registered_models(query, page_token=token2, max_results=20)
        assert token3 is not None
        assert result == rms[15:35]

        result, token4 = self._search_registered_models(query, page_token=token3, max_results=100)
        # assert that page token is None
        assert token4 is None
        assert result == rms[35:]

        # test that providing a completely invalid page token throws
        with pytest.raises(
            MlflowException, match=r"Invalid page token, could not base64-decode"
        ) as exception_context:
            self._search_registered_models(query, page_token="evilhax", max_results=20)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # test that providing too large of a max_results throws
        with pytest.raises(
            MlflowException, match=r"Invalid value for request parameter max_results"
        ) as exception_context:
            self._search_registered_models(query, page_token="evilhax", max_results=1e15)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_search_registered_model_order_by(self):
        rms = []
        # explicitly mock the creation_timestamps because timestamps seem to be unstable in Windows
        for i in range(50):
            with mock.patch(
                "mlflow.store.model_registry.sqlalchemy_store.get_current_time_millis",
                return_value=i,
            ):
                rms.append(self._rm_maker("RM{:03}".format(i)).name)

        # test flow with fixed max_results and order_by (test stable order across pages)
        returned_rms = []
        query = "name LIKE 'RM%'"
        result, token = self._search_registered_models(
            query, page_token=None, order_by=["name DESC"], max_results=5
        )
        returned_rms.extend(result)
        while token:
            result, token = self._search_registered_models(
                query, page_token=token, order_by=["name DESC"], max_results=5
            )
            returned_rms.extend(result)
        # name descending should be the opposite order of the current order
        assert rms[::-1] == returned_rms
        # last_updated_timestamp descending should have the newest RMs first
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["last_updated_timestamp DESC"], max_results=100
        )
        assert rms[::-1] == result
        # timestamp returns same result as last_updated_timestamp
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp DESC"], max_results=100
        )
        assert rms[::-1] == result
        # last_updated_timestamp ascending should have the oldest RMs first
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["last_updated_timestamp ASC"], max_results=100
        )
        assert rms == result
        # timestamp returns same result as last_updated_timestamp
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp ASC"], max_results=100
        )
        assert rms == result
        # timestamp returns same result as last_updated_timestamp
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp"], max_results=100
        )
        assert rms == result
        # name ascending should have the original order
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["name ASC"], max_results=100
        )
        assert rms == result
        # test that no ASC/DESC defaults to ASC
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["last_updated_timestamp"], max_results=100
        )
        assert rms == result
        with mock.patch(
            "mlflow.store.model_registry.sqlalchemy_store.get_current_time_millis", return_value=1
        ):
            rm1 = self._rm_maker("MR1").name
            rm2 = self._rm_maker("MR2").name
        with mock.patch(
            "mlflow.store.model_registry.sqlalchemy_store.get_current_time_millis", return_value=2
        ):
            rm3 = self._rm_maker("MR3").name
            rm4 = self._rm_maker("MR4").name
        query = "name LIKE 'MR%'"
        # test with multiple clauses
        result, _ = self._search_registered_models(
            query,
            page_token=None,
            order_by=["last_updated_timestamp ASC", "name DESC"],
            max_results=100,
        )
        assert result == [rm2, rm1, rm4, rm3]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp ASC", "name   DESC"], max_results=100
        )
        assert result == [rm2, rm1, rm4, rm3]
        # confirm that name ascending is the default, even if ties exist on other fields
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=[], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        # test default tiebreak with descending timestamps
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["last_updated_timestamp DESC"], max_results=100
        )
        assert result == [rm3, rm4, rm1, rm2]
        # test timestamp parsing
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp\tASC"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp\r\rASC"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp\nASC"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp  ASC"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        # validate order by key is case-insensitive
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp  asc"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp  aSC"], max_results=100
        )
        assert result == [rm1, rm2, rm3, rm4]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp  desc", "name desc"], max_results=100
        )
        assert result == [rm4, rm3, rm2, rm1]
        result, _ = self._search_registered_models(
            query, page_token=None, order_by=["timestamp  deSc", "name deSc"], max_results=100
        )
        assert result == [rm4, rm3, rm2, rm1]

    def test_search_registered_model_order_by_errors(self):
        query = "name LIKE 'RM%'"
        # test that invalid columns throw even if they come after valid columns
        with pytest.raises(
            MlflowException, match=r"Invalid order by key '.+' specified"
        ) as exception_context:
            self._search_registered_models(
                query,
                page_token=None,
                order_by=["name ASC", "creation_timestamp DESC"],
                max_results=5,
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test that invalid columns with random text throw even if they come after valid columns
        with pytest.raises(
            MlflowException, match=r"Invalid order_by clause '.+'"
        ) as exception_context:
            self._search_registered_models(
                query,
                page_token=None,
                order_by=["name ASC", "last_updated_timestamp DESC blah"],
                max_results=5,
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_set_model_version_tag(self):
        name1 = "SetModelVersionTag_TestMod"
        name2 = "SetModelVersionTag_TestMod 2"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        self._rm_maker(name1)
        self._rm_maker(name2)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        self._mv_maker(name1, "A/B", run_id_1, initial_tags)
        self._mv_maker(name1, "A/C", run_id_2, initial_tags)
        self._mv_maker(name2, "A/D", run_id_3, initial_tags)
        new_tag = ModelVersionTag("randomTag", "not a random value")
        self.store.set_model_version_tag(name1, 1, new_tag)
        all_tags = initial_tags + [new_tag]
        rm1mv1 = self.store.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in all_tags}

        # test overriding a tag with the same key
        overriding_tag = ModelVersionTag("key", "overriding")
        self.store.set_model_version_tag(name1, 1, overriding_tag)
        all_tags = [tag for tag in all_tags if tag.key != "key"] + [overriding_tag]
        rm1mv1 = self.store.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in all_tags}
        # does not affect other model versions with the same key
        rm1mv2 = self.store.get_model_version(name1, 2)
        rm2mv1 = self.store.get_model_version(name2, 1)
        assert rm1mv2.tags == {tag.key: tag.value for tag in initial_tags}
        assert rm2mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # can not set tag on deleted (non-existed) model version
        self.store.delete_model_version(name1, 2)
        with pytest.raises(
            MlflowException, match=rf"Model Version \(name={name1}, version=2\) not found"
        ) as exception_context:
            self.store.set_model_version_tag(name1, 2, overriding_tag)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # test cannot set tags that are too long
        long_tag = ModelVersionTag("longTagKey", "a" * 5001)
        with pytest.raises(
            MlflowException,
            match=r"Model version value '.+' had length \d+, which exceeded length limit of 5000",
        ) as exception_context:
            self.store.set_model_version_tag(name1, 1, long_tag)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # test can set tags that are somewhat long
        long_tag = ModelVersionTag("longTagKey", "a" * 4999)
        self.store.set_model_version_tag(name1, 1, long_tag)
        # can not set invalid tag
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            self.store.set_model_version_tag(name2, 1, ModelVersionTag(key=None, value=""))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name or version
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.set_model_version_tag(None, 1, ModelVersionTag(key="key", value="value"))
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Model version must be an integer"
        ) as exception_context:
            self.store.set_model_version_tag(
                name2, "I am not a version", ModelVersionTag(key="key", value="value")
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_delete_model_version_tag(self):
        name1 = "DeleteModelVersionTag_TestMod"
        name2 = "DeleteModelVersionTag_TestMod 2"
        initial_tags = [
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "some other value"),
        ]
        self._rm_maker(name1)
        self._rm_maker(name2)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        self._mv_maker(name1, "A/B", run_id_1, initial_tags)
        self._mv_maker(name1, "A/C", run_id_2, initial_tags)
        self._mv_maker(name2, "A/D", run_id_3, initial_tags)
        new_tag = ModelVersionTag("randomTag", "not a random value")
        self.store.set_model_version_tag(name1, 1, new_tag)
        self.store.delete_model_version_tag(name1, 1, "randomTag")
        rm1mv1 = self.store.get_model_version(name1, 1)
        assert rm1mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # testing deleting a key does not affect other model versions with the same key
        self.store.delete_model_version_tag(name1, 1, "key")
        rm1mv1 = self.store.get_model_version(name1, 1)
        rm1mv2 = self.store.get_model_version(name1, 2)
        rm2mv1 = self.store.get_model_version(name2, 1)
        assert rm1mv1.tags == {"anotherKey": "some other value"}
        assert rm1mv2.tags == {tag.key: tag.value for tag in initial_tags}
        assert rm2mv1.tags == {tag.key: tag.value for tag in initial_tags}

        # delete tag that is already deleted does nothing
        self.store.delete_model_version_tag(name1, 1, "key")
        rm1mv1 = self.store.get_model_version(name1, 1)
        assert rm1mv1.tags == {"anotherKey": "some other value"}

        # can not delete tag on deleted (non-existed) model version
        self.store.delete_model_version(name2, 1)
        with pytest.raises(
            MlflowException, match=rf"Model Version \(name={name2}, version=1\) not found"
        ) as exception_context:
            self.store.delete_model_version_tag(name2, 1, "key")
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # can not delete tag with invalid key
        with pytest.raises(MlflowException, match=r"Tag name cannot be None") as exception_context:
            self.store.delete_model_version_tag(name1, 2, None)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        # can not use invalid model name or version
        with pytest.raises(
            MlflowException, match=r"Registered model name cannot be empty"
        ) as exception_context:
            self.store.delete_model_version_tag(None, 2, "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        with pytest.raises(
            MlflowException, match=r"Model version must be an integer"
        ) as exception_context:
            self.store.delete_model_version_tag(name1, "I am not a version", "key")
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
