d4(o0).
d5(o9).
d12(o6).
d5(o4).
d14(o5).
d15(o5).
d7(o12).
d0(o4).
d10(o5).
d6(o0).
d1(o9).
d11(o12).
d4(o5).
d16(o2).
d11(o9).

d13(X) :- d14(X), not d12(X), not d5(X).
d7(X) :- d17(X), not d10(X), not d4(X).
d12(X) :- d9(X), not d1(X), not d17(X).
d1(X) :- not d6(X), d4(X), not d9(X).
d14(X) :- d17(X), not d10(X), not d15(X).
d8(X) :- not d6(X), not d7(X), d13(X).
d13(X) :- not d10(X), not d7(X), d15(X).
d14(X) :- d17(X), not d4(X), not d13(X).
d2(X) :- d0(X), not d15(X), not d13(X).
d3(X) :- not d1(X), not d2(X), d13(X).
d9(X) :- not d0(X), d3(X), not d14(X).
d17(X) :- not d6(X), not d8(X), d15(X).
d7(X) :- not d6(X), not d11(X), d13(X).
d6(X) :- d1(X), not d15(X), not d9(X).
d10(X) :- d8(X), not d12(X), not d14(X).