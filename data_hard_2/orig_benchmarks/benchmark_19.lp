d0(o1).
d8(o16).
d2(o14).
d14(o8).
d5(o14).
d6(o17).
d11(o3).
d10(o15).
d8(o4).
d6(o9).

d7(X) :- d6(X), d4(X), not d8(X).
d9(X) :- d10(X), not d14(X), d13(X).
d17(X) :- d8(X), d10(X), not d5(X).
d4(X) :- d6(X), d17(X), not d14(X).
d0(X) :- d5(X), d10(X), not d16(X).
d17(X) :- d8(X), d6(X), not d4(X).
d13(X) :- d17(X), not d14(X), d15(X).
d12(X) :- d9(X), not d8(X), d13(X).
d12(X) :- d5(X), not d16(X), d13(X).
d16(X) :- d7(X), d5(X), not d14(X).
d10(X) :- d5(X), d2(X), not d17(X).
d13(X) :- d10(X), d16(X), not d9(X).
d13(X) :- d5(X), d16(X), not d14(X).
d15(X) :- d8(X), d10(X), not d9(X).
d16(X) :- d5(X), not d4(X), d10(X).