#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o8).

#modeh(d5(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d5(o2)
}, {
  d5(o3), d5(o4), d5(o8)
}, {
  d1(o2).
  d2(o2).
  d3(o4).
  d3(o8).
  d4(o3).
}).