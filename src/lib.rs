use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
struct ProgramInput {
    c: Vec<f64>, // objective coefficents
    x: Vec<u8>, // variable type
    a: Vec<Vec<f64>>, // varible coefficents
    b: Vec<f64>, // rhs values
    e: Vec<u8>, // equality type: 0 = Less then, 1 = Equality, 2 = Greater then
    r#type: u8, // maximize (0) or minimize (1)
}

#[derive(Serialize, Debug)]
pub struct Step {
		pub tablue: Vec<Vec<String>>,
		pub step: usize,
		pub title: String,
		pub detail: String,
}

#[derive(Serialize, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub a: Vec<f64>,
    pub z: f64,
    pub solved: bool,
    pub steps: Vec<Step>,
}

impl Solution {
	pub fn new () -> Solution {
		Solution {
			x: Vec::new(),
			a: Vec::new(),
			s: Vec::new(),
			z: 0.0,
			solved: true,
			steps: Vec::new(),
		}
	}
}

#[derive(Debug)]
struct SolutionValue {
    value: f64,
		x: usize,
		y: Option<usize>
}

pub struct Program {
    tablue: Vec<Vec<f64>>,
		const_type_vec: Vec<u8>,
		nr_x_vars: usize,
		nr_s_vars: usize,
		nr_a_vars: usize,
		step_count: usize,
		solution: Solution,
}

impl Program {

    pub fn new () -> Program {
			Program {
				tablue: Vec::new(),
				const_type_vec: Vec::new(),
				nr_x_vars: 0,
				nr_s_vars: 0,
				nr_a_vars: 0,
				step_count: 0,
				solution: Solution::new()
			}
    }
    
    pub fn execute (mut self, json_problem: &String) -> String {
			let load_result = Program::load(&json_problem);
			println!["Request: {}",json_problem];
			let input: ProgramInput = match load_result {
				Ok( _ ) => load_result.unwrap(),
				Err( _ ) => return {
					println!("Error when trying to decode json to input");
					"Internal Error".to_string()
				},
			};

			self.setup_tablue(input);

			/*
			self.print_matrix();
			
			println!("{:?}", self.get_tablue_as_string());
			
			// println!("{:?}", x);
			let solution = self.get_solution();
			println!("{:?}", solution);
			*/

			self.get_solution_response()

    }

    fn load (json_problem: &String) -> Result<ProgramInput, serde_json::Error> {
			serde_json::from_str(&json_problem)
    }

    fn setup_tablue (&mut self, input: ProgramInput) {

			self.tablue = Vec::new();

			let nr_of_equal_const = input.e.iter().filter(|&n| *n == 2).count();
			let nr_of_greater_than_const = input.e.iter().filter(|&n| *n == 1).count();
			
			let len_a = input.a.len();
			let len_a0 = input.a[0].len();

			let artificial_const_values = vec![0.0;nr_of_greater_than_const + nr_of_equal_const];
			let mut artificial_goal_values = vec![1000000000.0;nr_of_greater_than_const + nr_of_equal_const];
			let slack = vec![0.0;len_a + nr_of_equal_const];
			let len_slack = slack.len();

			self.nr_x_vars = len_a0;
			self.nr_s_vars = len_slack;
			self.nr_a_vars = artificial_goal_values.len();

			// Setup top row
			self.tablue.push(input.c);
	
			self.tablue[0].append(&mut slack.clone());
			self.tablue[0].append(&mut artificial_goal_values);
			self.tablue[0].push(0.0);
			
			let mut e1 = 0;
			let mut e2 = 0;

			for i in 0..len_a {
				self.tablue.push(input.a[i].clone());
				self.tablue[i+1].append(&mut slack.clone());
				self.tablue[i+1].append(&mut artificial_const_values.clone());
				self.tablue[i+1].push(0.0);
				if input.e[i] == 0 {
					self.tablue[i+1][len_a0 + i - e2] = 1.0;
				}
				else if input.e[i] == 1 {
					self.tablue[i+1][len_a0 + i - e2] = -1.0;
					self.tablue[i+1][len_a0 + len_slack + e1] = 1.0;
					e1 += 1;
				}
				else if input.e[i] == 2 {
					self.tablue[i+1][len_a0 + len_slack + e1] = 1.0;
					e1 += 1;
					e2 += 1;
				}
				else {
					self.tablue[i+1][len_a0 + i] = 1.0
				}
				let len_tablue = self.tablue.len();
				let len_tablue_sub = self.tablue[i+1].len();
				self.tablue[i+1][len_tablue_sub - 1] = input.b[len_tablue - 2]
			}

			self.setup_x(input.x, len_a, nr_of_equal_const);

			if len_a - nr_of_equal_const > 0 {
				let mut tablue = self.get_tablue();
				for row in &mut tablue {
					row.truncate(self.nr_x_vars + self.nr_s_vars);
				}
				self.step_count = self.step_count + 1;
				let mut detail = String::new();
				for i in 1..self.nr_s_vars + 1 {
					if i > 1 {
						detail.push_str(", ");
					}
					detail.push_str(&format!( "s{}", i ));
				}
				let step = Step {
					tablue: tablue,
					step: self.step_count,
					title: "Added slackvariables".to_string(),
					detail: detail,
				};
				self.solution.steps.push(step);
			}

			// println!("{:?}", serde_json::to_string(&tablue).unwrap());
			// serde_json::to_string(&tablue).unwrap()

			let mut x_extend = vec![3; self.nr_a_vars];
			self.const_type_vec.append(&mut x_extend);
			
			if nr_of_greater_than_const + nr_of_equal_const > 0 {
				self.step_count = self.step_count + 1;
				let mut detail = String::new();
				for i in 1..(nr_of_greater_than_const + nr_of_equal_const + 1) {
					if i > 1 {
						detail.push_str(", ");
					}
					detail.push_str(&format!( "a{}", i ));
				}
				let step = Step {
					tablue: self.get_tablue(),
					step: self.step_count,
					title: "Added artificial variables".to_string(),
					detail: detail,
				};
				self.solution.steps.push(step);
			}

			if nr_of_greater_than_const + nr_of_equal_const > 0 {
				self.update_objective_function();
				let step = Step {
					tablue: self.get_tablue(),
					step: self.step_count,
					title: "Updated objective row".to_string(),
					detail: "<p class='text-danger'> z always calulated outside pivot operations</p>".to_string(),
				};
				self.solution.steps.push(step);
			}
		}
		
		fn update_objective_function (&mut self) {
			for _row in 1..self.tablue.len() {
				let _start = self.nr_x_vars + self.nr_s_vars;
				
			}
			/*
			for row in range(1,len(self.tablue)):
				start = len(self.A[0]) + len(self.slack)
				if self.tablue[row][start:-1].count(1) > 0:
					self.tablue[0] = [self.tablue[0][i] - self.tablue[row][i] * 1000000000 for i in range(len(self.tablue[0]))]
					self.tablue[0][-1] = 0
			*/
		}

    fn setup_x (&mut self, const_type_vec: Vec<u8>, len_a: usize, nr_of_equal_const: usize) {
        self.const_type_vec = const_type_vec;
        let mut x_extend = vec![2;len_a + nr_of_equal_const];
        self.const_type_vec.append(&mut x_extend);
		}
		
		/*
		fn get_solution(&self) -> Vec<SolutionValue> {
			let mut solution_values: Vec<SolutionValue> = Vec::new();
			let len_tablue = self.tablue.len() - 1;
			let len_tablue_zero = self.tablue[0].len() - 1;
			for j in 0..len_tablue_zero {

				let mut i = 0;
				let mut ok = true;
				let mut x: Option<usize> = None;
				while ok && i <= len_tablue {
					if self.tablue[i][j] == 1.0 && x.is_none() {
						x = Some(i);
					}
					else if self.tablue[i][j] != 0.0 {
						ok = false;
					}
					i = i + 1;
				}

				if ok {
					match x {
						Some(i) => {
							let y = self.tablue[i].len() - 1;
							let value = self.tablue[i][y];
							let solution = SolutionValue {
								value: value,
								x: j,
								y: Some(i)
							};
							solution_values.push(solution);
						},
						None => {
							let solution = SolutionValue {
								value: 0.0,
								x: j,
								y: None
							};
							solution_values.push(solution);
						}
					}
				}
			}
			solution_values
		}
		*/

		fn get_tablue (&self) -> Vec<Vec<String>> {
			let mut tablue = Vec::new();
			let mut variable_vec = Vec::new();
			let mut var_nr = 0;
			let mut var_type = "x";
			for _ in 0..self.const_type_vec.len() {
				var_nr = var_nr + 1;
				let value = format!( "{}{}", var_type, var_nr );
				variable_vec.push(value);
				// Reset from "x" to "s"
				if self.nr_x_vars == var_nr && var_type == "x" {
					var_type = "s";
					var_nr = 0;
				}
				// Reset from "s" to "a"
				if self.nr_s_vars == var_nr && var_type == "s" {
					var_type = "a";
					var_nr = 0;
				}
			}
			variable_vec.push("z, b".to_string());
			tablue.push(variable_vec);
			for row in self.tablue.iter() {
				let row2: Vec<_> = row.iter().map(|float| format!("{}", float)).collect();
				tablue.push(row2);
			}
			tablue
		}

		/*
		fn print_matrix (&mut self) {
      println!("{}", self.tablue.iter().fold("", |tab, col| { print!("{}{:?}", tab, col); "\n" }) );
		}
		*/

    fn get_solution_response (&self) -> String {
			let mut response = r#"{"solution":"#.to_string();
			response.push_str(&serde_json::to_string(&self.solution).unwrap());
			response.push_str("}");
			response
    }

}


#[cfg(test)]
mod tests {
    use crate::{Program, ProgramInput};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn input_ok() {
        let json_string = r#"{"c":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
        let result: Result<ProgramInput, serde_json::Error> = serde_json::from_str(&json_string);
        assert!(result.is_ok());
		}
		
		#[test]
    fn input_ok_with_ints() {
        let json_string = r#"{"c":[-4,1],"x":[1,0],"a":[[7,-2],[0,1],[2,-2]],"b":[14,3,3],"e":[0,0,0],"type":0}"#;
        let result: Result<ProgramInput, serde_json::Error> = serde_json::from_str(&json_string);
        assert!(result.is_ok());
    }

    #[test]
    fn input_err() {
        let json_string = r#"{"cc":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
        let result: Result<ProgramInput, serde_json::Error> = serde_json::from_str(&json_string);
        assert!(result.is_err());
    }

    #[test]
    fn load_1_nr_of_equal_const() {
        let json_string = r#"{"c":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
        let input = Program::load(&json_string.to_string()).unwrap();
        let nr_of_equal_const = input.e.iter().filter(|&n| *n == 1).count();
        assert!(nr_of_equal_const == 0);
    }

    #[test]
    fn load_2_nr_of_less() {
        let json_string = r#"{"c":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
        let input = Program::load(&json_string.to_string()).unwrap();
        let nr_of_less_then_const = input.e.iter().filter(|&n| *n == 0).count();
        assert!(nr_of_less_then_const == 3);
    }

    #[test]
    fn load_3_number_of_constrains() {
        let json_string = r#"{"c":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
        let input = Program::load(&json_string.to_string()).unwrap();
        let number_of_constrains = input.a.len();
        assert!(number_of_constrains == 3);
    }
    
    #[test]
    fn execute_ok() {
        let json_string = r#"{"c":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
				let program = Program::new();
				let result = program.execute(&json_string.to_string());
        assert!(result != "Internal Error");
    }

    #[test]
    fn execute_err() {
        let json_string = r#"{"cd":[-4.0,1.0],"x":[1,0],"a":[[7.0,-2.0],[0.0,1.0],[2.0,-2.0]],"b":[14.0,3.0,3.0],"e":[0,0,0],"type":0}"#;
				let program = Program::new();
				let result = program.execute(&json_string.to_string());
        assert!(result == "Internal Error");
    }

}

/*


	def updateObjectiveFunction(self):
		for row in range(1,len(self.tablue)):
			start = len(self.A[0]) + len(self.slack)
			if self.tablue[row][start:-1].count(1) > 0:
				self.tablue[0] = [self.tablue[0][i] - self.tablue[row][i] * 1000000000 for i in range(len(self.tablue[0]))]
				self.tablue[0][-1] = 0


	def solve(self):

		self.success = True

		self.steps = []
		self.step = 1
		self.setupTablue()

		q = 1
		solved = False

		while not solved:

			while any(round(i,5) < 0 for i in self.tablue[0][:-1]) and q < 1000 and self.success:

				if self.getPrimalPivotIndexs() == False:
					self.steps.append({"step":self.step,"text":"The problem is unbounded", "s": [], "type":0, "matrix": self.getTablue()})
					self.success = False
					continue

				pivotCol, pivotRow = self.getPrimalPivotIndexs()
				self.pivot(pivotRow, pivotCol)

				self.updateZ()

				self.steps.append({"step":self.step,"text":"Pivot", "row": pivotRow, "column": pivotCol + 1, "type": 1, "matrix": self.getTablue()})
				self.step += 1

				q += 1

			self.getSolution()

			if not self.success:
				solved = True
			elif self.checkInfeasibility():
				solved = True
			elif self.isSolved():
				solved = True
			elif q >= 1000:
				solved = True
				self.success = False
				self.steps.append({"step":self.step,"text":"Failed to many steps", "s": [], "type":0, "matrix": self.getTablue()})
			else:
				self.addCuts()

			q += 1


	def getPrimalPivotIndexs(self):

		pivotCol = self.tablue[0][:-1].index(min(self.tablue[0][:-1]))

		t = []

		for i in range(1,len(self.tablue)):
			if self.tablue[i][pivotCol] != 0 and float(self.tablue[i][-1])/self.tablue[i][pivotCol] > 0 and self.tablue[i][-1] >= 0:
				t.append(float(self.tablue[i][-1])/self.tablue[i][pivotCol])
			else:
				t.append(1000000000)

		pivotRow = t.index(min(t)) + 1

		if sum(i >= 1000000000 for i in t) == len(t):
			return False
		else:
			return pivotCol, pivotRow


	def possiblePivot(self,pivotCol):
		if self.tablue[0][pivotCol] < 0:
			t = []
			for i in range(len(self.b)):
				if self.tablue[i + 1][pivotCol] != 0 and float(self.tablue[i + 1][-1])/self.tablue[i + 1][pivotCol] > 0 and self.tablue[i][-1] >= 0:
					t.append(float(self.tablue[i + 1][-1])/self.tablue[i + 1][pivotCol])
				else:
					t.append(1000000000)
			pivotRow = t.index(min(t)) + 1

			if sum(i >= 1000000000 for i in t) != len(t):
				return 1
		return 0


	def pivot(self, pivotRow, pivotCol):
		try:
			self.tablue[pivotRow] = [x / float(self.tablue[pivotRow][pivotCol]) for x in self.tablue[pivotRow]]

			for row in range(len(self.tablue)):
				if row != pivotRow and self.tablue[row][pivotCol] != 0:
					subAmount = self.tablue[row][pivotCol]
					self.tablue[row] = [self.tablue[row][col] - subAmount * float(self.tablue[pivotRow][col]) for col in range(len(self.tablue[row]))]

		except:
			self.success = False
			self.steps.append({"step":self.step,"text":"Failed numbers got to small", "s": [], "type":0, "matrix": self.getTablue()})

	def addCuts(self):

		maxCuts = 2
		cuts = 0

		if len(self.tablue) >= len(self.A) * 10:
			self.success = False
			self.steps.append({"step":self.step,"text":"Failed to many cuts", "s": [], "type":0, "matrix": self.getTablue()})

		else:
			for i in range(len(self.solution)):
				if cuts < maxCuts and len(self.tablue) < len(self.A) * 10:
					if self.x[self.solution[i][1]] == 1:
						if self.solution[i][2] >= 0 and not round(Decimal(self.solution[i][0]),4).is_integer():
							if self.e[self.solution[i][2]-1] == 1:
								self.addGomoryMinMIRCut(i)
								cuts += 1
							else:
								self.addGomoryMaxMIRCut(i)
								cuts += 1
			self.makeFeasable()


	def addGomoryMaxMIRCut(self, i):

		cut = self.tablue[self.solution[i][2]][:]
		cut[-1] = -float(cut[-1] - math.floor(cut[-1]))

		for j in range(len(cut)-1):
			if j < len(self.x) and self.x[j] == 1:
				cut[j] = -float(cut[j] - math.floor(cut[j]))
				if cut[j] <= cut[-1]:
					cut[j] = cut[-1] * (1 + cut[j])/(1 + cut[-1])
			else:
				cut[j] = -cut[j]
				if cut[j] >= 0:
					cut[j] = cut[j] * cut[-1] / (1 + cut[-1])

		cut.insert(-1, 1)
		self.x.append(2)

		for i in self.tablue:
			i.insert(-1, 0)

		self.tablue.append(cut)
		self.e.append(0)

		cutText = self.cutToText(cut)
		self.steps.append({"step":self.step,"text":"Added max cut", "cut": cutText,"type": 2, "matrix": self.getTablue()})
		self.step += 1


	def addGomoryMinMIRCut(self, i):

		cut = self.tablue[self.solution[i][2]][:]
		cut[-1] = float(cut[-1] - math.ceil(cut[-1]))

		for j in range(len(cut)-1):
			if j < len(self.x) and self.x[j] == 1:
				cut[j] = -float(math.ceil(cut[j]) - cut[j])
				if cut[j] <= cut[-1]:
					cut[j] = -cut[-1] * (1 + cut[j])/(1 + cut[-1])
			else:
				cut[j] = cut[j]
				if cut[j] >= 0:
					cut[j] = cut[j] * cut[-1] / (1 + cut[-1])

		cut.insert(-1, 1)
		self.x.append(2)

		for i in range(len(self.tablue)):
			self.tablue[i].insert(-1, 0)

		self.tablue.append(cut)
		self.e.append(1)

		cutText = self.cutToText(cut)
		self.steps.append({"step":self.step,"text":"Added min cut", "cut": cutText,"type": 2, "matrix": self.getTablue()})
		self.step += 1


	def makeFeasable(self):

		unfeasible = True
		k = 0

		while unfeasible and self.success:

			k += 1
			minimum = 0
			pivotRow = 0

			for i in range(1, len(self.tablue)):
				if minimum > round(self.tablue[i][-1],7):
					minimum = self.tablue[i][-1]
					pivotRow = i

			if pivotRow == 0:
				unfeasible = False
			elif k >= 50:
				unfeasible = False
				self.success = False
				self.steps.append({"step":self.step,"text":"Failed on dual pivot", "s": [], "type":0, "matrix": self.getTablue()})
			else:
				pivotCol = self.getDualPivotCol(pivotRow)
				self.pivot(pivotRow,pivotCol)

				self.updateZ()
				self.steps.append({"step":self.step,"text":"Pivot, dual", "row": pivotRow, "column": pivotCol + 1, "type": 1, "matrix": self.getTablue()})
				self.step += 1


	def getDualPivotCol(self, pivotRow):
		t = []

		for i in range(len(self.tablue[0])-1):

			if self.tablue[pivotRow][i] != 0 and float(-self.tablue[0][i])/self.tablue[pivotRow][i] > 0:
				t.append(float(-self.tablue[0][i])/self.tablue[pivotRow][i])
			else:
				t.append(1000000000)

		return t.index(min(t))


	def getSolution(self):

		self.solution = []

		for j in range(len(self.tablue[0])-1):

			i = 0
			ok = True
			x = -1

			while ok and i <= len(self.tablue) - 1:
				if self.tablue[i][j] == 1 and x == -1:
					x = i
				elif self.tablue[i][j] != 0:
					ok = False
				i += 1

			if x > 0 and ok:
				self.solution.append((self.tablue[x][-1], j, x))
			else:
				self.solution.append((0, j, -1))


	def isSolved(self):
		Solved = True
		j = 0
		while Solved and j < len(self.tablue[0]) - 1:
			value = self.solution[j][0]
			valueUp = math.ceil(value*10000)/10000
			valueDown = math.floor(value*10000)/10000
			if not float(valueUp).is_integer() and not float(valueDown).is_integer() and self.x[j] == 1:
				Solved = False
			j += 1

		return Solved

	def checkInfeasibility(self):
		Infeasible = False
		j = 0
		while not Infeasible and j < len(self.tablue[0]) - 1:
			value = self.solution[j][0]
			if self.x[j] == 3 and value != 0:
				Infeasible = True
				self.success = False
				self.steps.append({"step":self.step,"text":"The problem is infeasible", "s": [], "type":0, "matrix": self.getTablue()})
			j += 1

		return Infeasible


	def cutToText(self, cut):

		if(self.x[-1] == 2):
			cutText = "s%i " % (self.x.count(2))
		elif(self.x[-1] == 3):
			cutText = "a%i " % (self.x.count(3))

		for i in range(len(cut)):
			value = cut[i]

			if cut[i] != 0 and (i < len(cut) - 2):

				if cut[i] > 0:
					cutText = cutText + "+ "
				elif cut[i] < 0:
					cutText = cutText + "- "
					value = value * -1

				if float(value).is_integer():
					if not value == 1:
						cutText = cutText + "%i" % value
				else:
					cutText = cutText + str(round(value, 3))

				if i < len(self.c):
					cutText = cutText + " x%i " % (i + 1)
				else:
					if(self.x[i] == 2):
						cutText = cutText + "s%i " % (self.x[:i+1].count(2))
					elif(self.x[i] == 3):
						cutText = cutText + "a%i " % (self.x[:i+1].count(3))

			elif len(cutText) > 0 and i == len(cut) - 1:

				cutText = cutText + "= "

				if value < 0:
					cutText = cutText + "- "
					value = value * -1

				if float(cut[i]).is_integer():
					cutText = cutText + ("%i" % value)
				else:
					cutText = cutText + str(round(value, 3))

		return cutText


	def updateZ(self):
		self.getSolution()
		Z = 0
		j = 0
		for i in self.solution:
			if j < len(self.c):
				Z += i[0] * - self.c[j]
			j += 1

		if self.type == 1:
			self.tablue[0][-1] = -Z
		else:
			self.tablue[0][-1] = Z


	def getTablue(self):
		matrix = [[]]
		j = 0
		for i in self.x:
			j += 1
			if i < 2:
				matrix[0].append("x%i" % (self.x[0:j].count(1) + self.x[0:j].count(0)))
			elif i == 2:
				matrix[0].append("s%i" % (self.x[0:j].count(i)))
			elif i == 3:
				matrix[0].append("a%i" % (self.x[0:j].count(i)))

		matrix[0].append("z, b")
		matrix.extend(copy.deepcopy(self.tablue))

		return matrix


	def printTablue(self):
		tablue = []
		for i in self.tablue:
			tablue.append(str(i))
		return tablue


	def printSolution(self):
		Z = 0
		j = 0
		text = ""

		for i in self.solution:
			text = text + str("x%i = %.2f " % (i[1] + 1, i[0]))
			if j < len(self.c):
				Z += i[0] * - self.c[j]
			j += 1

		return text + str("Z  = %.2f" % Z)


	def getJSONsolution(self):

		solution = {"x":[], "s":[], "a":[]}

		Z = 0
		j = 0

		for i in self.solution:
			if self.x[i[1]] < 2:
				solution["x"].append(float("%.2f" % i[0]))
			elif self.x[i[1]] == 2:
				solution["s"].append(float("%.2f" % i[0]))
			else:
				solution["a"].append(float("%.2f" % i[0]))
			if j < len(self.c):
				Z += i[0] * - self.c[j]
			j += 1

		solution["z"] = float("%.2f" % Z)
		if self.type == 1:
			solution["z"] = solution["z"] * -1
		solution["solved"] = self.success
		solution["steps"] = self.steps
		jsonSolution = {"solution": solution}

		return json.dumps(jsonSolution, separators=(',',':'))


if __name__ == "__main__":
	# os.system('clear')

	try:
		jsonString = sys.argv[1]
		j = json.loads(jsonString)
	except:
		jsonString = '{"c":[-4,1],"x":[1,1],"A":[[7,-2],[0,1],[2,-2]],"b":[14,3,3],"e":[0,1,0],"type":0}'

	y = Solver(jsonString)
	y.solve()
	x = y.getJSONsolution()
	print x
*/