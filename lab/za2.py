from mpmath import mp

mp.dps = 1_000_000
my_birthdate = "060802"

print("Searching for my birthdate in digits of constants")
print("My birthdate is: 06/08/2002")
print("Throwing out the slashes and shortening the year to just two digits, it is:", my_birthdate)
print("===================================================================================")

pi_no_year = str(mp.pi)[2:].find(my_birthdate[:-2])
pi_year = str(mp.pi)[2:].find(my_birthdate)

print("Birthday in digits of pi:")
print(f"\t -> without year: {pi_no_year + 1}")
print(f"\t -> with year: {pi_year + 1}")

e_no_year = str(mp.e)[2:].find(my_birthdate[:-2])
e_year = str(mp.e)[2:].find(my_birthdate)

print("\nBirthday in digits of e:")
print(f"\t -> without year: {e_no_year + 1}")
print(f"\t -> with year: {e_year + 1}")

sqrt2_no_year = str(mp.sqrt(2))[2:].find(my_birthdate[:-2])
sqrt2_year = str(mp.sqrt(2))[2:].find(my_birthdate)

print("\nBirthday in digits of sqrt(2):")
print(f"\t -> without year: {sqrt2_no_year + 1}")
print(f"\t -> with year: {sqrt2_year + 1}")

print("===================================================================================")